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
from threading import Lock

from loguru import logger
from openai import OpenAI
from tenacity import RetryCallState, retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

from lib.llm import client as default_client
from lib.personalized_data import (
    PersonalizedSample,
    build_personalized_samples,
    dataset_stats,
)

client: OpenAI = default_client


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
) -> list[dict]:
    messages: list[dict] = []
    if context_mode == "task":
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


def _expected_ids(samples: list[PersonalizedSample]) -> set[str]:
    return {
        f"{sample.user}__{sample.target_task}__{os.path.basename(session.file_path)}__turn_{turn_idx}"
        for sample in samples
        for session in sample.target_sessions
        for turn_idx in range(session.assistant_turns)
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
    temperature: float,
    max_tokens: int,
    timeout: int,
    finished_ids: set[str],
    context_mode: str,
) -> list[dict]:
    records: list[dict] = []
    for session, session_file, turn_idx, prefix, source_reply in _iter_static_turns(sample):
        sample_id = f"{sample.user}__{sample.target_task}__{session_file}__turn_{turn_idx}"
        if sample_id in finished_ids:
            continue
        messages = _build_replay_messages(
            profile=sample.profile,
            task_context=session.task_context,
            dialogue_prefix=prefix,
            context_mode=context_mode,
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
) -> None:
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    finished_ids = load_finished_ids(output_jsonl)
    expected = _expected_ids(samples)
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
        "--replay_context_mode",
        type=str,
        default="raw",
        choices=["raw", "task", "profile"],
        help=(
            "Candidate-visible replay context. raw uses only the original "
            "dialogue prefix; task additionally injects task context; profile "
            "injects user profile and task context. Default raw is the benchmark setting."
        ),
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
        args.output_jsonl = f"outputs/static_replay/{model_tag}_{args.split}_responses.jsonl"

    logger.info(f"Candidate model: {args.model}")
    logger.info(f"Backend: {'custom @ ' + args.base_url if args.base_url else 'default OpenAI API'}")
    logger.info(f"Replay context mode: {args.replay_context_mode}")
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
    )


if __name__ == "__main__":
    main()
