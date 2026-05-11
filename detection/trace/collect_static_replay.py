"""
Static replay response collection.

For each target assistant turn in the personalized benchmark, this script
keeps the original dialogue prefix before that assistant turn, asks a candidate
LLM to generate one reply, and writes the generated response to JSONL.

This is a single-turn static replay. Generated responses are not rolled into
later turns, so every candidate model is evaluated on the same prefixes.
"""

from __future__ import annotations

import json
import os
import traceback
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from dataclasses import dataclass
from threading import Lock

from loguru import logger
from openai import OpenAI
from tenacity import RetryCallState, retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

from lib.llm import client as default_client
from lib.dialogue_memory import (
    DialogueMemoryIndex,
    DialogueMemoryRecord,
    format_dialogue_memory_prompt,
)
from lib.personalized_data import (
    PersonalizedSample,
    build_personalized_samples,
    dataset_stats,
)

client: OpenAI = default_client


@dataclass(frozen=True)
class ReplayTurnSelection:
    sample_id: str
    selection_mode: str
    selection_score: float
    selection_reasons: list[str]


class EmptyCandidateResponse(RuntimeError):
    def __init__(self, message: str, response_payload: dict) -> None:
        super().__init__(message)
        self.response_payload = response_payload


def _to_jsonable(obj: object) -> object:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, list):
        return [_to_jsonable(item) for item in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "model_dump"):
        try:
            return _to_jsonable(obj.model_dump())
        except Exception:
            pass
    return str(obj)


def _strip_model_wrappers(text: str) -> str:
    out = text.strip()
    if out.startswith("<think>") and "</think>" in out:
        out = out.split("</think>", 1)[1].strip()
    if out.startswith("```") and out.endswith("```"):
        lines = out.splitlines()
        if len(lines) >= 2:
            out = "\n".join(lines[1:-1]).strip()
    return out


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


def _safe_name(text: str, limit: int = 160) -> str:
    return "".join(
        c if c.isalnum() or c in {"_", "-", "."} else "_"
        for c in text
    )[:limit]


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
        payload["http_headers"] = dict(getattr(response, "headers", {}) or {})
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
    response_payload = getattr(e, "response_payload", None)
    if response_payload is not None:
        payload["llm_response"] = response_payload
    return payload


def _dump_generation_failure(
    sample_id: str,
    model: str,
    messages: list[dict],
    e: Exception,
) -> None:
    dump_dir = "outputs/static_replay/generation_failures"
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
        "Retrying candidate generation: attempt={}, error_type={}, error={}",
        retry_state.attempt_number,
        type(exc).__name__ if exc else "unknown",
        repr(exc),
    )


def _format_profile(profile: dict) -> str:
    gender = "女" if profile.get("gender") == "Female" else "男"
    return (
        f"性别：{gender}\n"
        f"年龄：{profile.get('age', '未知')}\n"
        f"职业：{profile.get('occupation', '未知')}\n"
        f"背景：{profile.get('background', '未知')}\n"
        f"性格：{'，'.join(profile.get('personality', []))}\n"
        f"兴趣：{'，'.join(profile.get('daily_interests', []))}"
    )


def _build_replay_messages(
    profile: dict,
    task_context: str,
    dialogue_prefix: list[dict],
    context_mode: str,
    dialogue_memories: list[DialogueMemoryRecord] | None = None,
    dialogue_memory_max_chars_per_item: int = 700,
) -> list[dict]:
    messages: list[dict] = []
    if context_mode in {"dialogue_memory_tfidf", "dialogue_memory_diverse"}:
        memory_prompt = format_dialogue_memory_prompt(
            dialogue_memories or [],
            max_chars_per_item=dialogue_memory_max_chars_per_item,
        )
        if memory_prompt:
            messages.append({"role": "system", "content": memory_prompt})
    elif context_mode == "task":
        messages.append({
            "role": "system",
            "content": (
                "Continue the conversation by answering the user's latest "
                "message. Use the task context only as background. Do not "
                "mention that you are being evaluated.\n\n"
                f"【任务背景】\n{task_context}"
            ),
        })
    elif context_mode == "profile":
        messages.append({
            "role": "system",
            "content": (
                "Continue the conversation by answering the user's latest "
                "message. Use the user profile and task context only as "
                "background. Do not mention that you are being evaluated.\n\n"
                f"【用户画像】\n{_format_profile(profile)}\n\n"
                f"【任务背景】\n{task_context}"
            ),
        })
    messages.extend({"role": u["role"], "content": u["content"]} for u in dialogue_prefix)
    return messages


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
    choice = response.choices[0]
    message = choice.message
    content = _message_content_to_text(getattr(message, "content", ""))
    if content:
        return _strip_model_wrappers(content)
    response_payload = {
        "id": getattr(response, "id", None),
        "model": getattr(response, "model", None),
        "created": getattr(response, "created", None),
        "usage": _to_jsonable(getattr(response, "usage", None)),
        "choice": {
            "finish_reason": getattr(choice, "finish_reason", None),
            "index": getattr(choice, "index", None),
            "message": _to_jsonable(message),
        },
    }
    refusal = getattr(message, "refusal", "")
    raise EmptyCandidateResponse(refusal or "empty candidate response", response_payload)


def _iter_static_turns(sample: PersonalizedSample):
    for session in sample.target_sessions:
        session_file = os.path.basename(session.file_path)
        prefix: list[dict] = []
        assistant_idx = 0
        for utt in session.history:
            if utt["role"] == "assistant":
                yield session, session_file, assistant_idx, list(prefix), utt["content"]
                assistant_idx += 1
            prefix.append({"role": utt["role"], "content": utt["content"]})


def _expected_ids(
    samples: list[PersonalizedSample],
    selected_ids: set[str] | None = None,
) -> set[str]:
    ids = {
        f"{sample.user}__{sample.target_task}__{os.path.basename(session.file_path)}__turn_{turn_idx}"
        for sample in samples
        for session in sample.target_sessions
        for turn_idx in range(session.assistant_turns)
    }
    if selected_ids is not None:
        ids &= selected_ids
    return ids


def _last_user_text(prefix: list[dict]) -> str:
    for message in reversed(prefix):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _is_substantive_turn(prefix: list[dict], source_reply: str) -> bool:
    user_text = _last_user_text(prefix).strip()
    reply = source_reply.strip()
    if len(user_text) < 8:
        return False
    ack_phrases = {
        "好的",
        "好",
        "可以",
        "没问题",
        "明白",
        "收到",
        "当然",
        "当然可以",
    }
    compact = reply.replace("。", "").replace("！", "").replace("!", "").strip()
    if compact in ack_phrases:
        return False
    if len(reply) < 25 and len(user_text) < 25:
        return False
    return True


def _has_specific_dsat_reason(score: int | None, reason: str | None) -> bool:
    if score is None or score > 3:
        return False
    reason_text = (reason or "").strip()
    return reason_text not in {"", "满意", "其它", "其他", "无", "none", "None"}


def _score_replay_turn(
    score: int | None,
    reason: str | None,
    prefix: list[dict],
    source_reply: str,
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    selection_score = 0.0

    if score is not None and score <= 3:
        selection_score += 3.0
        reasons.append("score_le_3")
    if score in {3, 4}:
        selection_score += 2.0
        reasons.append("boundary_3_4")
    if _has_specific_dsat_reason(score, reason):
        selection_score += 1.5
        reasons.append("specific_dsat_reason")

    if _is_substantive_turn(prefix, source_reply):
        selection_score += 1.0
        reasons.append("substantive")
    else:
        selection_score -= 1.5
        reasons.append("non_substantive")

    if len(_last_user_text(prefix).strip()) >= 20:
        selection_score += 1.0
        reasons.append("clear_user_request")

    if score == 5:
        selection_score -= 0.5
        reasons.append("positive_control_candidate")

    return selection_score, reasons


def _build_replay_selection(
    samples: list[PersonalizedSample],
    selection_mode: str,
    hard_max_per_block: int,
    hard_max_per_session: int,
    hard_global_budget: int,
    hard_positive_controls_per_block: int,
    hard_min_turn_idx: int,
    hard_score_quota: dict[int, int] | None = None,
    hard_min_per_user: int = 1,
) -> dict[str, ReplayTurnSelection] | None:
    if selection_mode == "full":
        return None

    selected: dict[str, ReplayTurnSelection] = {}
    all_candidates: list[tuple[str, str, str, int | None, float, list[str]]] = []
    score_counter: Counter[int | None] = Counter()
    reason_counter: Counter[str] = Counter()

    for sample in samples:
        block_id = sample.block_id
        by_session: dict[str, list[tuple[str, int | None, float, list[str]]]] = defaultdict(list)
        for session, session_file, turn_idx, prefix, source_reply in _iter_static_turns(sample):
            if turn_idx < hard_min_turn_idx:
                continue
            score = (
                int(session.satisfaction_scores[turn_idx])
                if turn_idx < len(session.satisfaction_scores)
                else None
            )
            reason = (
                session.dissatisfaction_reasons[turn_idx]
                if turn_idx < len(session.dissatisfaction_reasons)
                else None
            )
            selection_score, reasons = _score_replay_turn(score, reason, prefix, source_reply)
            if "non_substantive" in reasons and score is not None and score >= 4:
                continue
            sample_id = f"{sample.user}__{sample.target_task}__{session_file}__turn_{turn_idx}"
            by_session[session_file].append((sample_id, score, selection_score, reasons))

        session_limited: list[tuple[str, int | None, float, list[str]]] = []
        for turns in by_session.values():
            turns = sorted(turns, key=lambda x: (-x[2], x[0]))
            session_limited.extend(turns[:hard_max_per_session])

        low_or_boundary = [
            t for t in session_limited
            if t[1] is not None and int(t[1]) <= 4
        ]
        positive_controls = [
            t for t in session_limited
            if t[1] == 5 and "substantive" in t[3]
        ]
        low_or_boundary = sorted(low_or_boundary, key=lambda x: (-x[2], x[0]))
        positive_controls = sorted(positive_controls, key=lambda x: (-x[2], x[0]))

        block_selected = low_or_boundary[:hard_max_per_block]
        remaining = hard_max_per_block - len(block_selected)
        if remaining > 0 and hard_positive_controls_per_block > 0:
            block_selected.extend(
                positive_controls[:min(remaining, hard_positive_controls_per_block)]
            )

        for sample_id, score, selection_score, reasons in block_selected:
            all_candidates.append((sample_id, block_id, sample.target_task, score, selection_score, reasons))

    all_candidates = sorted(all_candidates, key=lambda x: (-x[4], x[1], x[0]))
    if hard_global_budget > 0 and hard_min_per_user > 0:
        by_user: dict[str, list[tuple[str, str, str, int | None, float, list[str]]]] = defaultdict(list)
        for candidate in all_candidates:
            user = candidate[1].split("__", 1)[0]
            by_user[user].append(candidate)
        warm_selected: list[tuple[str, str, str, int | None, float, list[str]]] = []
        warm_used_ids: set[str] = set()
        for user in sorted(by_user):
            for candidate in by_user[user][:hard_min_per_user]:
                if len(warm_selected) >= hard_global_budget:
                    break
                warm_selected.append(candidate)
                warm_used_ids.add(candidate[0])
            if len(warm_selected) >= hard_global_budget:
                break
        all_candidates = warm_selected + [
            candidate for candidate in all_candidates
            if candidate[0] not in warm_used_ids
        ]

    if hard_global_budget > 0 and hard_score_quota:
        by_score: dict[int | None, list[tuple[str, str, str, int | None, float, list[str]]]] = defaultdict(list)
        for candidate in all_candidates:
            by_score[candidate[3]].append(candidate)

        quota_selected: list[tuple[str, str, str, int | None, float, list[str]]] = []
        used_ids: set[str] = set()
        if hard_min_per_user > 0:
            by_user: set[str] = set()
            for candidate in all_candidates:
                user = candidate[1].split("__", 1)[0]
                if user in by_user:
                    continue
                quota_selected.append(candidate)
                used_ids.add(candidate[0])
                by_user.add(user)
                if len(quota_selected) >= hard_global_budget:
                    break
        for score, quota in sorted(hard_score_quota.items()):
            if quota <= 0:
                continue
            picked: list[tuple[str, str, str, int | None, float, list[str]]] = []
            for candidate in by_score.get(score, []):
                if candidate[0] in used_ids:
                    continue
                picked.append(candidate)
                if len(picked) >= quota:
                    break
            quota_selected.extend(picked)
            used_ids.update(candidate[0] for candidate in picked)

        remaining_budget = hard_global_budget - len(quota_selected)
        if remaining_budget > 0:
            for candidate in all_candidates:
                if candidate[0] in used_ids:
                    continue
                quota_selected.append(candidate)
                used_ids.add(candidate[0])
                if len(quota_selected) >= hard_global_budget:
                    break
        all_candidates = quota_selected[:hard_global_budget]
    elif hard_global_budget > 0:
        all_candidates = all_candidates[:hard_global_budget]

    for sample_id, _, _, score, selection_score, reasons in all_candidates:
        selected[sample_id] = ReplayTurnSelection(
            sample_id=sample_id,
            selection_mode=selection_mode,
            selection_score=selection_score,
            selection_reasons=reasons,
        )
        score_counter[score] += 1
        reason_counter.update(reasons)

    logger.info(
        "Static replay selection: mode={}, selected_turns={}, score_dist={}, reason_dist={}",
        selection_mode,
        len(selected),
        dict(sorted(score_counter.items(), key=lambda x: (x[0] is None, x[0]))),
        dict(reason_counter.most_common()),
    )
    return selected


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
    temperature: float,
    max_tokens: int,
    timeout: int,
    finished_ids: set[str],
    context_mode: str,
    selection_mode: str,
    selected_turns: dict[str, ReplayTurnSelection] | None,
    dialogue_memory_top_k: int,
    dialogue_memory_max_chars_per_item: int,
    dialogue_memory_local_history_size: int,
) -> list[dict]:
    records: list[dict] = []
    use_dialogue_memory = context_mode in {
        "dialogue_memory_tfidf",
        "dialogue_memory_diverse",
    }
    dialogue_memory_index = (
        DialogueMemoryIndex(
            user=sample.user,
            sessions=sample.history_sessions,
            local_history_size=dialogue_memory_local_history_size,
        )
        if use_dialogue_memory
        else None
    )
    for session, session_file, turn_idx, prefix, source_reply in _iter_static_turns(sample):
        sample_id = f"{sample.user}__{sample.target_task}__{session_file}__turn_{turn_idx}"
        selection = selected_turns.get(sample_id) if selected_turns is not None else None
        if selected_turns is not None and selection is None:
            continue
        if sample_id in finished_ids:
            continue
        dialogue_memories: list[DialogueMemoryRecord] = []
        if dialogue_memory_index is not None:
            strategy = (
                "diverse"
                if context_mode == "dialogue_memory_diverse"
                else "tfidf"
            )
            dialogue_memories = dialogue_memory_index.retrieve(
                dialogue_prefix=prefix,
                k=dialogue_memory_top_k,
                strategy=strategy,
            )
        messages = _build_replay_messages(
            profile=sample.profile,
            task_context=session.task_context,
            dialogue_prefix=prefix,
            context_mode=context_mode,
            dialogue_memories=dialogue_memories,
            dialogue_memory_max_chars_per_item=dialogue_memory_max_chars_per_item,
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
                _dump_generation_failure(sample_id, model, messages, e)
            except Exception as dump_err:
                logger.warning(f"Failed to dump generation failure for {sample_id}: {dump_err}")
            details = _exception_payload(e)
            logger.error(
                f"Generation failed: {sample_id}: "
                f"{details['error_type']}: {details['repr']}; "
                f"dump=outputs/static_replay/generation_failures/{_safe_name(sample_id)}.json"
            )
            continue
        records.append({
            "sample_id": sample_id,
            "user": sample.user,
            "target_task": sample.target_task,
            "target_file": session_file,
            "turn_idx": turn_idx,
            "candidate_model": model,
            "replay_context_mode": context_mode,
            "selection_mode": selection_mode,
            "selection_score": selection.selection_score if selection else None,
            "selection_reasons": selection.selection_reasons if selection else [],
            "dialogue_memory_top_k": dialogue_memory_top_k if use_dialogue_memory else 0,
            "dialogue_memory_records": [
                memory.to_metadata(max_chars=240)
                for memory in dialogue_memories
            ],
            "task_context": session.task_context,
            "dialogue_prefix": prefix,
            "candidate_response": candidate_response,
            "source_assistant_reply": source_reply,
            "source_chat_model": session.chat_model,
            "gold_score": session.satisfaction_scores[turn_idx]
            if turn_idx < len(session.satisfaction_scores) else None,
            "gold_reason": session.dissatisfaction_reasons[turn_idx]
            if turn_idx < len(session.dissatisfaction_reasons) else None,
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
    temperature: float,
    max_tokens: int,
    timeout: int,
    context_mode: str,
    selection_mode: str,
    selected_turns: dict[str, ReplayTurnSelection] | None,
    dialogue_memory_top_k: int,
    dialogue_memory_max_chars_per_item: int,
    dialogue_memory_local_history_size: int,
) -> None:
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    finished_ids = load_finished_ids(output_jsonl)
    selected_ids = set(selected_turns) if selected_turns is not None else None
    expected = _expected_ids(samples, selected_ids=selected_ids)
    logger.info(f"Already finished turns: {len(finished_ids & expected)} / {len(expected)}")
    output_lock = Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                collect_sample,
                sample,
                model,
                temperature,
                max_tokens,
                timeout,
                finished_ids,
                context_mode,
                selection_mode,
                selected_turns,
                dialogue_memory_top_k,
                dialogue_memory_max_chars_per_item,
                dialogue_memory_local_history_size,
            ): sample
            for sample in samples
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="blocks"):
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


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Collect static replay candidate responses")
    parser.add_argument("--model", type=str, required=True, help="Candidate LLM name")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "all"])
    parser.add_argument("--train_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--target_tasks", type=str, nargs="+", default=None)
    parser.add_argument("--min_history_sessions", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--limit_users", type=int, default=0)
    parser.add_argument("--user_offset", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output_jsonl", type=str, default="")
    parser.add_argument("--base_url", type=str, default="", help="OpenAI-compatible API base URL")
    parser.add_argument("--api_key", type=str, default="", help="API key for the selected endpoint")
    parser.add_argument(
        "--selection_mode",
        type=str,
        default="full",
        choices=["full", "hard", "filter"],
        help=(
            "Turn selection mode. full replays every assistant turn; hard/filter "
            "selects high-value turns with per-session and per-block caps."
        ),
    )
    parser.add_argument(
        "--hard_max_per_block",
        type=int,
        default=3,
        help="Max selected turns per user-task block in hard/filter mode.",
    )
    parser.add_argument(
        "--hard_max_per_session",
        type=int,
        default=1,
        help="Max selected turns per dialogue session in hard/filter mode.",
    )
    parser.add_argument(
        "--hard_global_budget",
        type=int,
        default=300,
        help="Global selected-turn cap in hard/filter mode; <=0 disables the cap.",
    )
    parser.add_argument(
        "--hard_positive_controls_per_block",
        type=int,
        default=1,
        help="Max score-5 positive-control turns per block in hard/filter mode.",
    )
    parser.add_argument(
        "--hard_min_turn_idx",
        type=int,
        default=1,
        help=(
            "Minimum 0-based assistant turn index selected in hard/filter mode. "
            "Default 1 skips the first assistant reply in each session."
        ),
    )
    parser.add_argument(
        "--hard_score_quota",
        type=str,
        default="1:25,2:50,4:75",
        help=(
            "Comma-separated minimum score quotas in hard/filter mode, e.g. "
            "'1:25,2:50,4:75'. Remaining budget is filled by selection score."
        ),
    )
    parser.add_argument(
        "--hard_min_per_user",
        type=int,
        default=1,
        help=(
            "Minimum selected turns per user in hard/filter mode when candidates "
            "are available. Default 1 improves user coverage."
        ),
    )
    parser.add_argument(
        "--replay_context_mode",
        type=str,
        default="raw",
        choices=[
            "raw",
            "task",
            "profile",
            "dialogue_memory_tfidf",
            "dialogue_memory_diverse",
        ],
        help=(
            "Candidate-visible replay context. raw uses only the original "
            "dialogue prefix; task additionally injects task context; profile "
            "injects user profile and task context; dialogue_memory_* injects "
            "unlabeled cross-scenario dialogue memories. Default raw is the benchmark setting."
        ),
    )
    parser.add_argument(
        "--dialogue_memory_top_k",
        type=int,
        default=4,
        help="Number of retrieved unlabeled dialogue memories for dialogue_memory_* modes.",
    )
    parser.add_argument(
        "--dialogue_memory_max_chars_per_item",
        type=int,
        default=700,
        help="Max assistant-reply characters per retrieved dialogue memory in the prompt.",
    )
    parser.add_argument(
        "--dialogue_memory_local_history_size",
        type=int,
        default=4,
        help="Number of previous messages stored per dialogue-memory record.",
    )
    return parser


def main() -> None:
    global client
    args = parse_args().parse_args()
    if args.base_url:
        client = OpenAI(base_url=args.base_url, api_key=args.api_key or "EMPTY")

    samples = build_personalized_samples(
        split=args.split,
        train_ratio=args.train_ratio,
        seed=args.split_seed,
        min_history_sessions=args.min_history_sessions,
        target_tasks=args.target_tasks,
    )
    samples = _select_user_subset(samples, args.limit_users, args.user_offset)
    if args.limit > 0:
        samples = samples[:args.limit]

    if not args.output_jsonl:
        model_tag = args.model.replace("/", "_").replace(":", "_")
        selection_tag = "" if args.selection_mode == "full" else f"_{args.selection_mode}"
        args.output_jsonl = (
            f"outputs/static_replay/{model_tag}_{args.split}"
            f"{selection_tag}_responses.jsonl"
        )

    hard_score_quota: dict[int, int] = {}
    if args.hard_score_quota.strip():
        for item in args.hard_score_quota.split(","):
            if not item.strip():
                continue
            score_text, quota_text = item.split(":", 1)
            hard_score_quota[int(score_text)] = int(quota_text)

    selected_turns = _build_replay_selection(
        samples=samples,
        selection_mode=args.selection_mode,
        hard_max_per_block=args.hard_max_per_block,
        hard_max_per_session=args.hard_max_per_session,
        hard_global_budget=args.hard_global_budget,
        hard_positive_controls_per_block=args.hard_positive_controls_per_block,
        hard_min_turn_idx=args.hard_min_turn_idx,
        hard_score_quota=hard_score_quota,
        hard_min_per_user=args.hard_min_per_user,
    )

    logger.info(f"Candidate model: {args.model}")
    logger.info(f"Backend: {'custom @ ' + args.base_url if args.base_url else 'default OpenAI API'}")
    logger.info(f"Replay context mode: {args.replay_context_mode}")
    if args.replay_context_mode.startswith("dialogue_memory_"):
        logger.info(
            "Dialogue memory: top_k={}, max_chars_per_item={}, local_history_size={}",
            args.dialogue_memory_top_k,
            args.dialogue_memory_max_chars_per_item,
            args.dialogue_memory_local_history_size,
        )
    logger.info(f"Selection mode: {args.selection_mode}")
    if selected_turns is not None:
        logger.info(
            "Selection caps: max_per_block={}, max_per_session={}, global_budget={}, positive_controls_per_block={}",
            args.hard_max_per_block,
            args.hard_max_per_session,
            args.hard_global_budget,
            args.hard_positive_controls_per_block,
        )
        logger.info(f"Selection min turn idx: {args.hard_min_turn_idx}")
        logger.info(f"Selection score quota: {hard_score_quota}")
        logger.info(f"Selection min per user: {args.hard_min_per_user}")
    logger.info(f"Output: {args.output_jsonl}")
    logger.info(f"Dataset stats: {dataset_stats(samples)}")

    collect_all(
        samples=samples,
        model=args.model,
        output_jsonl=args.output_jsonl,
        max_workers=args.max_workers,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        context_mode=args.replay_context_mode,
        selection_mode=args.selection_mode,
        selected_turns=selected_turns,
        dialogue_memory_top_k=args.dialogue_memory_top_k,
        dialogue_memory_max_chars_per_item=args.dialogue_memory_max_chars_per_item,
        dialogue_memory_local_history_size=args.dialogue_memory_local_history_size,
    )


if __name__ == "__main__":
    main()
