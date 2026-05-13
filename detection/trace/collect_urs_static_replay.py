"""Collect URS session-level static replay responses."""

from __future__ import annotations

import json
import os
import traceback
from argparse import ArgumentParser
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock
from typing import Literal

from loguru import logger
from openai import OpenAI
from tenacity import RetryCallState, retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

from lib.dialogue_memory import (
    DialogueMemoryIndex,
    DialogueMemoryRecord,
    format_dialogue_memory_prompt,
)
from lib.llm import client as default_client
from lib.personalized_data import PersonalizedSample, SessionData
from lib.urs_data import (
    URS_INTENT_LIST,
    build_urs_personalized_samples,
    urs_dataset_stats,
)

client: OpenAI = default_client

ReplayContextMode = Literal["raw", "dialogue_memory_tfidf", "dialogue_memory_diverse"]
ReplayGranularity = Literal["first_user", "last_user"]


@dataclass(frozen=True)
class UrsReplaySelection:
    sample_id: str
    selection_score: float
    selection_reasons: list[str]


class EmptyCandidateResponse(RuntimeError):
    pass


def _safe_name(text: str, limit: int = 160) -> str:
    return "".join(c if c.isalnum() or c in {"_", "-", "."} else "_" for c in text)[:limit]


def _message_content_to_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p).strip()
    return str(content)


def _strip_model_wrappers(text: str) -> str:
    out = text.strip()
    if out.startswith("<think>") and "</think>" in out:
        out = out.split("</think>", 1)[1].strip()
    if out.startswith("```") and out.endswith("```"):
        lines = out.splitlines()
        if len(lines) >= 2:
            out = "\n".join(lines[1:-1]).strip()
    return out


def _exception_payload(e: Exception) -> dict:
    payload = {
        "error_type": type(e).__name__,
        "error": str(e),
        "repr": repr(e),
        "traceback": "".join(traceback.format_exception_only(type(e), e)).strip(),
        "full_traceback": traceback.format_exc(),
    }
    response = getattr(e, "response", None)
    if response is not None:
        payload["http_status_code"] = getattr(response, "status_code", None)
        text = getattr(response, "text", "")
        if text:
            payload["http_response_text"] = text[:4000]
        try:
            payload["http_response_json"] = response.json()
        except Exception:
            pass
    body = getattr(e, "body", None)
    if body is not None:
        payload["error_body"] = body
    return payload


def _dump_generation_failure(
    sample_id: str,
    model: str,
    messages: list[dict],
    e: Exception,
) -> None:
    dump_dir = "outputs/urs_static_replay/generation_failures"
    os.makedirs(dump_dir, exist_ok=True)
    prefix = os.path.join(dump_dir, _safe_name(sample_id))
    meta = {
        "sample_id": sample_id,
        "model": model,
        "messages_count": len(messages),
        "messages_chars": sum(len(str(m.get("content", ""))) for m in messages),
        **_exception_payload(e),
    }
    with open(prefix + ".json", "w", encoding="utf-8") as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=2)
    with open(prefix + ".messages.json", "w", encoding="utf-8") as fp:
        json.dump(messages, fp, ensure_ascii=False, indent=2)


def _log_retry_sleep(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "Retrying URS replay generation: attempt={}, error_type={}, error={}",
        retry_state.attempt_number,
        type(exc).__name__ if exc else "unknown",
        repr(exc),
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    before_sleep=_log_retry_sleep,
    reraise=True,
)
def _generate_response(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    content = _message_content_to_text(response.choices[0].message.content)
    if not content:
        raise EmptyCandidateResponse("empty candidate response")
    return _strip_model_wrappers(content)


def _session_file(session: SessionData) -> str:
    return os.path.basename(session.file_path)


def _sample_id(sample: PersonalizedSample, session: SessionData) -> str:
    return f"{sample.user}__{sample.target_task}__{_session_file(session)}__turn_0"


def _first_user_prefix(session: SessionData) -> list[dict]:
    for utt in session.history:
        if utt.get("role") == "user":
            return [{"role": "user", "content": utt.get("content", "")}]
    return []


def _last_user_prefix(session: SessionData) -> list[dict]:
    last_assistant_idx = -1
    for idx, utt in enumerate(session.history):
        if utt.get("role") == "assistant":
            last_assistant_idx = idx
    if last_assistant_idx <= 0:
        return _first_user_prefix(session)
    return [
        {"role": utt.get("role", ""), "content": utt.get("content", "")}
        for utt in session.history[:last_assistant_idx]
        if utt.get("role") in {"user", "assistant"}
    ]


def _replay_prefix(session: SessionData, granularity: ReplayGranularity) -> list[dict]:
    if granularity == "first_user":
        return _first_user_prefix(session)
    if granularity == "last_user":
        return _last_user_prefix(session)
    raise ValueError(f"Unknown replay granularity: {granularity}")


def _build_messages(
    prefix: list[dict],
    context_mode: ReplayContextMode,
    memories: list[DialogueMemoryRecord],
    max_chars_per_item: int,
) -> list[dict]:
    messages: list[dict] = []
    if context_mode in {"dialogue_memory_tfidf", "dialogue_memory_diverse"}:
        memory_prompt = format_dialogue_memory_prompt(
            memories,
            max_chars_per_item=max_chars_per_item,
        )
        if memory_prompt:
            messages.append({"role": "system", "content": memory_prompt})
    messages.extend({"role": m["role"], "content": m["content"]} for m in prefix)
    return messages


def _score_session_for_selection(session: SessionData) -> tuple[float, list[str]]:
    score = int(session.satisfaction_scores[0])
    reasons: list[str] = []
    selection_score = 0.0
    if score <= 3:
        selection_score += 3.0
        reasons.append("score_le_3")
    if score in {3, 4}:
        selection_score += 2.0
        reasons.append("boundary_3_4")
    if score == 5:
        selection_score -= 0.5
        reasons.append("positive_control_candidate")
    if len(session.history) >= 2:
        selection_score += 1.0
        reasons.append("has_assistant_reply")
    first_user = _first_user_prefix(session)
    if first_user and len(first_user[0]["content"].strip()) >= 20:
        selection_score += 1.0
        reasons.append("clear_initial_request")
    return selection_score, reasons


def _build_selection(
    samples: list[PersonalizedSample],
    selection_mode: str,
    global_budget: int,
    max_per_user: int,
    score_quota: dict[int, int],
) -> dict[str, UrsReplaySelection] | None:
    if selection_mode == "full":
        return None

    candidates: list[tuple[str, str, int, float, list[str]]] = []
    for sample in samples:
        for session in sample.target_sessions:
            if not session.satisfaction_scores:
                continue
            sid = _sample_id(sample, session)
            score = int(session.satisfaction_scores[0])
            selection_score, reasons = _score_session_for_selection(session)
            candidates.append((sid, sample.user, score, selection_score, reasons))
    candidates.sort(key=lambda x: (-x[3], x[1], x[0]))

    if max_per_user > 0:
        per_user: Counter[str] = Counter()
        limited: list[tuple[str, str, int, float, list[str]]] = []
        overflow: list[tuple[str, str, int, float, list[str]]] = []
        for item in candidates:
            if per_user[item[1]] < max_per_user:
                limited.append(item)
                per_user[item[1]] += 1
            else:
                overflow.append(item)
        candidates = limited + overflow

    selected: list[tuple[str, str, int, float, list[str]]] = []
    used: set[str] = set()
    if score_quota:
        by_score: dict[int, list[tuple[str, str, int, float, list[str]]]] = defaultdict(list)
        for item in candidates:
            by_score[item[2]].append(item)
        for score, quota in sorted(score_quota.items()):
            for item in by_score.get(score, [])[:quota]:
                if item[0] in used:
                    continue
                selected.append(item)
                used.add(item[0])

    for item in candidates:
        if global_budget > 0 and len(selected) >= global_budget:
            break
        if item[0] in used:
            continue
        selected.append(item)
        used.add(item[0])

    if global_budget > 0:
        selected = selected[:global_budget]

    score_dist = Counter(item[2] for item in selected)
    logger.info(
        "URS static replay selection: mode={}, selected={}, score_dist={}",
        selection_mode,
        len(selected),
        dict(sorted(score_dist.items())),
    )
    return {
        sid: UrsReplaySelection(
            sample_id=sid,
            selection_score=selection_score,
            selection_reasons=reasons,
        )
        for sid, _, _, selection_score, reasons in selected
    }


def load_finished_ids(output_jsonl: str) -> set[str]:
    if not os.path.exists(output_jsonl):
        return set()
    finished: set[str] = set()
    with open(output_jsonl, "r", encoding="utf-8") as fp:
        for line in fp:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            finished.add(obj.get("sample_id", ""))
    return finished


def collect_sample(
    sample: PersonalizedSample,
    model: str,
    context_mode: ReplayContextMode,
    granularity: ReplayGranularity,
    selected: dict[str, UrsReplaySelection] | None,
    finished_ids: set[str],
    temperature: float,
    max_tokens: int,
    timeout: int,
    dialogue_memory_top_k: int,
    dialogue_memory_max_chars_per_item: int,
    dialogue_memory_local_history_size: int,
) -> list[dict]:
    records: list[dict] = []
    use_memory = context_mode in {"dialogue_memory_tfidf", "dialogue_memory_diverse"}
    memory_index = (
        DialogueMemoryIndex(
            user=sample.user,
            sessions=sample.history_sessions,
            local_history_size=dialogue_memory_local_history_size,
        )
        if use_memory else None
    )
    for session in sample.target_sessions:
        sid = _sample_id(sample, session)
        selection = selected.get(sid) if selected is not None else None
        if selected is not None and selection is None:
            continue
        if sid in finished_ids:
            continue
        prefix = _replay_prefix(session, granularity)
        if not prefix:
            logger.warning(f"Skipping empty replay prefix: {sid}")
            continue
        memories: list[DialogueMemoryRecord] = []
        if memory_index is not None:
            memories = memory_index.retrieve(
                dialogue_prefix=prefix,
                k=dialogue_memory_top_k,
                strategy="diverse" if context_mode == "dialogue_memory_diverse" else "tfidf",
            )
        messages = _build_messages(
            prefix=prefix,
            context_mode=context_mode,
            memories=memories,
            max_chars_per_item=dialogue_memory_max_chars_per_item,
        )
        try:
            candidate_response = _generate_response(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except Exception as e:
            try:
                _dump_generation_failure(sid, model, messages, e)
            except Exception as dump_err:
                logger.warning(f"Failed to dump generation failure for {sid}: {dump_err}")
            logger.error(f"Generation failed: {sid}: {type(e).__name__}: {e}")
            continue

        replay_history = list(prefix) + [{"role": "assistant", "content": candidate_response}]
        records.append({
            "sample_id": sid,
            "dataset": "urs",
            "user": sample.user,
            "target_task": sample.target_task,
            "target_file": _session_file(session),
            "candidate_model": model,
            "replay_context_mode": context_mode,
            "replay_granularity": granularity,
            "selection_mode": "full" if selected is None else "hard",
            "selection_score": selection.selection_score if selection else None,
            "selection_reasons": selection.selection_reasons if selection else [],
            "task_context": session.task_context,
            "dialogue_prefix": prefix,
            "candidate_response": candidate_response,
            "replay_history": replay_history,
            "source_session_history": session.history,
            "source_chat_model": session.chat_model,
            "gold_score": session.satisfaction_scores[0] if session.satisfaction_scores else None,
            "gold_reason": session.dissatisfaction_reasons[0] if session.dissatisfaction_reasons else None,
            "dialogue_memory_top_k": dialogue_memory_top_k if use_memory else 0,
            "dialogue_memory_records": [
                memory.to_metadata(max_chars=240) for memory in memories
            ],
        })
    return records


def _select_user_subset(
    samples: list[PersonalizedSample],
    limit_users: int,
    user_offset: int,
) -> list[PersonalizedSample]:
    if limit_users <= 0:
        return samples
    users = list(dict.fromkeys(s.user for s in samples))
    selected = set(users[user_offset:user_offset + limit_users])
    return [s for s in samples if s.user in selected]


def collect_all(
    samples: list[PersonalizedSample],
    model: str,
    output_jsonl: str,
    max_workers: int,
    context_mode: ReplayContextMode,
    granularity: ReplayGranularity,
    selected: dict[str, UrsReplaySelection] | None,
    temperature: float,
    max_tokens: int,
    timeout: int,
    dialogue_memory_top_k: int,
    dialogue_memory_max_chars_per_item: int,
    dialogue_memory_local_history_size: int,
) -> None:
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    finished_ids = load_finished_ids(output_jsonl)
    logger.info(f"Already finished sessions: {len(finished_ids)}")
    output_lock = Lock()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                collect_sample,
                sample,
                model,
                context_mode,
                granularity,
                selected,
                finished_ids,
                temperature,
                max_tokens,
                timeout,
                dialogue_memory_top_k,
                dialogue_memory_max_chars_per_item,
                dialogue_memory_local_history_size,
            ): sample
            for sample in samples
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="urs_replay_blocks"):
            sample = futures[future]
            try:
                records = future.result()
            except Exception as e:
                logger.error(f"Block {sample.block_id} failed: {e}")
                continue
            if not records:
                continue
            with output_lock:
                with open(output_jsonl, "a", encoding="utf-8") as fp:
                    for record in records:
                        fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def _parse_score_quota(text: str) -> dict[int, int]:
    out: dict[int, int] = {}
    if not text.strip():
        return out
    for item in text.split(","):
        if not item.strip():
            continue
        score_text, quota_text = item.split(":", 1)
        out[int(score_text)] = int(quota_text)
    return out


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Collect URS session-level static replay responses")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "all"])
    parser.add_argument("--train_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--languages", type=str, nargs="+", default=["zh", "en"], choices=["zh", "en"])
    parser.add_argument("--target_intents", type=str, nargs="+", default=None, choices=URS_INTENT_LIST)
    parser.add_argument("--min_history_sessions", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--limit_users", type=int, default=0)
    parser.add_argument("--user_offset", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output_jsonl", type=str, default="")
    parser.add_argument("--selection_mode", type=str, default="hard", choices=["full", "hard"])
    parser.add_argument("--hard_global_budget", type=int, default=300)
    parser.add_argument("--hard_max_per_user", type=int, default=4)
    parser.add_argument("--hard_score_quota", type=str, default="1:25,2:50,3:75,4:75")
    parser.add_argument(
        "--replay_context_mode",
        type=str,
        default="raw",
        choices=["raw", "dialogue_memory_tfidf", "dialogue_memory_diverse"],
    )
    parser.add_argument("--replay_granularity", type=str, default="first_user", choices=["first_user", "last_user"])
    parser.add_argument("--dialogue_memory_top_k", type=int, default=4)
    parser.add_argument("--dialogue_memory_max_chars_per_item", type=int, default=700)
    parser.add_argument("--dialogue_memory_local_history_size", type=int, default=4)
    return parser


def main() -> None:
    global client
    args = parse_args().parse_args()
    if args.base_url:
        client = OpenAI(base_url=args.base_url, api_key=args.api_key or "EMPTY")

    samples = build_urs_personalized_samples(
        split=args.split,
        train_ratio=args.train_ratio,
        seed=args.split_seed,
        min_history_sessions=args.min_history_sessions,
        target_intents=args.target_intents,
        languages=tuple(args.languages),
    )
    samples = _select_user_subset(samples, args.limit_users, args.user_offset)
    if args.limit > 0:
        samples = samples[:args.limit]

    if not args.output_jsonl:
        model_tag = args.model.replace("/", "_").replace(":", "_")
        lang_tag = "_".join(args.languages)
        args.output_jsonl = (
            f"outputs/urs_static_replay/{model_tag}_{args.split}_{lang_tag}_"
            f"{args.selection_mode}_{args.replay_context_mode}_{args.replay_granularity}_responses.jsonl"
        )

    selected = _build_selection(
        samples=samples,
        selection_mode=args.selection_mode,
        global_budget=args.hard_global_budget,
        max_per_user=args.hard_max_per_user,
        score_quota=_parse_score_quota(args.hard_score_quota),
    )

    logger.info(f"Candidate model: {args.model}")
    logger.info(f"Backend: {'custom @ ' + args.base_url if args.base_url else 'default OpenAI API'}")
    logger.info(f"Replay context mode: {args.replay_context_mode}")
    logger.info(f"Replay granularity: {args.replay_granularity}")
    logger.info(f"Selection mode: {args.selection_mode}")
    logger.info(f"Output: {args.output_jsonl}")
    logger.info(f"Dataset stats: {urs_dataset_stats(samples)}")

    collect_all(
        samples=samples,
        model=args.model,
        output_jsonl=args.output_jsonl,
        max_workers=args.max_workers,
        context_mode=args.replay_context_mode,  # type: ignore[arg-type]
        granularity=args.replay_granularity,  # type: ignore[arg-type]
        selected=selected,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        dialogue_memory_top_k=args.dialogue_memory_top_k,
        dialogue_memory_max_chars_per_item=args.dialogue_memory_max_chars_per_item,
        dialogue_memory_local_history_size=args.dialogue_memory_local_history_size,
    )


if __name__ == "__main__":
    main()
