"""Debug one URS static replay sample and dump the raw chat response."""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser

from openai import OpenAI

import trace.collect_urs_static_replay as replay
from lib.dialogue_memory import DialogueMemoryIndex
from lib.urs_data import build_urs_personalized_samples


def _find_sample(sample_id: str, split: str, languages: list[str]):
    samples = build_urs_personalized_samples(split=split, languages=tuple(languages))
    for sample in samples:
        for session in sample.target_sessions:
            sid = replay._sample_id(sample, session)
            if sid == sample_id:
                return sample, session
    raise RuntimeError(f"sample_id not found: {sample_id}")


def _to_jsonable(obj: object) -> object:
    return replay._to_jsonable(obj)


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Debug one URS static replay sample.")
    parser.add_argument("--sample_id", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "all"])
    parser.add_argument("--languages", type=str, nargs="+", default=["zh", "en"], choices=["zh", "en"])
    parser.add_argument(
        "--replay_context_mode",
        type=str,
        default="dialogue_memory_diverse",
        choices=["raw", "dialogue_memory_tfidf", "dialogue_memory_diverse"],
    )
    parser.add_argument("--replay_granularity", type=str, default="first_user", choices=["first_user", "last_user"])
    parser.add_argument("--dialogue_memory_top_k", type=int, default=4)
    parser.add_argument("--dialogue_memory_max_chars_per_item", type=int, default=700)
    parser.add_argument("--dialogue_memory_local_history_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output_json", type=str, default="")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key) if args.base_url else replay.client
    sample, session = _find_sample(args.sample_id, args.split, args.languages)
    prefix = replay._replay_prefix(session, args.replay_granularity)
    memories = []
    if args.replay_context_mode in {"dialogue_memory_tfidf", "dialogue_memory_diverse"}:
        index = DialogueMemoryIndex(
            user=sample.user,
            sessions=sample.history_sessions,
            local_history_size=args.dialogue_memory_local_history_size,
        )
        memories = index.retrieve(
            dialogue_prefix=prefix,
            k=args.dialogue_memory_top_k,
            strategy="diverse" if args.replay_context_mode == "dialogue_memory_diverse" else "tfidf",
        )
    messages = replay._build_messages(
        prefix=prefix,
        context_mode=args.replay_context_mode,
        memories=memories,
        max_chars_per_item=args.dialogue_memory_max_chars_per_item,
    )
    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    choice = response.choices[0]
    message = choice.message
    payload = {
        "sample_id": args.sample_id,
        "request": {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
            "replay_context_mode": args.replay_context_mode,
            "replay_granularity": args.replay_granularity,
        },
        "messages": messages,
        "retrieved_memories": [memory.to_metadata(max_chars=500) for memory in memories],
        "response": {
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "created": getattr(response, "created", None),
            "usage": _to_jsonable(getattr(response, "usage", None)),
            "choice": {
                "finish_reason": getattr(choice, "finish_reason", None),
                "index": getattr(choice, "index", None),
                "message": _to_jsonable(message),
            },
        },
    }
    output_json = args.output_json or (
        "outputs/urs_static_replay/debug/"
        + replay._safe_name(args.sample_id)
        + f"__{args.replay_context_mode}.json"
    )
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    print(json.dumps(payload["response"], ensure_ascii=False, indent=2))
    print(f"Saved: {output_json}")


if __name__ == "__main__":
    main()
