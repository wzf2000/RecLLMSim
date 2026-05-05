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
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from loguru import logger
from openai import OpenAI
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

from lib.llm import client as default_client
from lib.personalized_data import (
    PersonalizedSample,
    build_personalized_samples,
    dataset_stats,
)

client: OpenAI = default_client


def _strip_model_wrappers(text: str) -> str:
    out = text.strip()
    if out.startswith("<think>") and "</think>" in out:
        out = out.split("</think>", 1)[1].strip()
    if out.startswith("```") and out.endswith("```"):
        lines = out.splitlines()
        if len(lines) >= 2:
            out = "\n".join(lines[1:-1]).strip()
    return out


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
) -> list[dict]:
    system = (
        "You are a helpful assistant. Continue the conversation by answering "
        "the user's latest message. Use the provided user profile and task "
        "context only as background. Do not mention that you are being evaluated.\n\n"
        "请根据以下用户画像和任务背景继续对话，只输出助手回复本身。\n\n"
        f"【用户画像】\n{_format_profile(profile)}\n\n"
        f"【任务背景】\n{task_context}"
    )
    messages: list[dict] = [{"role": "system", "content": system}]
    messages.extend({"role": u["role"], "content": u["content"]} for u in dialogue_prefix)
    return messages


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    before_sleep=before_sleep_log(logger, log_level=40),
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
    ).choices[0].message
    if response.content:
        return _strip_model_wrappers(response.content)
    raise RuntimeError(response.refusal or "empty candidate response")


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
            logger.error(f"Generation failed: {sample_id}: {e}")
            continue
        records.append({
            "sample_id": sample_id,
            "user": sample.user,
            "target_task": sample.target_task,
            "target_file": session_file,
            "turn_idx": turn_idx,
            "candidate_model": model,
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
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--output_jsonl", type=str, default="")
    parser.add_argument("--base_url", type=str, default="", help="OpenAI-compatible API base URL")
    parser.add_argument("--api_key", type=str, default="", help="API key for the selected endpoint")
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
    )


if __name__ == "__main__":
    main()
