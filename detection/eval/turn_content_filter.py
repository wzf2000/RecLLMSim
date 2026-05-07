"""
LLM-based turn content annotation and grouped evaluation.

This script separates assistant turns that contain substantive task content from
turns that are mostly clarifying questions, acknowledgements, or meta replies.

Typical workflow from detection/:

  python eval/turn_content_filter.py annotate \
      --result_file outputs/personalized/history_prior_delta_v3_none_n3_limit20.jsonl \
      --output_jsonl outputs/personalized/turn_content_annotations_u20.jsonl \
      --vllm_base_url http://localhost:8000/v1 \
      --model Qwen/Qwen3-8B

  python eval/turn_content_filter.py analyze \
      --annotations_jsonl outputs/personalized/turn_content_annotations_u20.jsonl \
      --result_files hpd_v2=outputs/personalized/history_prior_delta_v2_none_n3.jsonl \
                     hpd_v3=outputs/personalized/history_prior_delta_v3_none_n3_limit20.jsonl \
      --output_json outputs/personalized/turn_content_compare_u20.json
"""

from __future__ import annotations

import json
import math
import os
import re
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, TypeVar

from loguru import logger
from openai import OpenAI
from pydantic import BaseModel
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

from lib.llm import client as default_client
from lib.personalized_data import PersonalizedSample, SessionData, build_personalized_samples


T = TypeVar("T", bound=BaseModel)
client: OpenAI = default_client


class TurnContentAnnotation(BaseModel):
    has_substantive_content: bool
    content_type: Literal[
        "substantive_answer",
        "mixed_answer",
        "clarifying_question",
        "ack_or_meta",
        "other",
    ]
    task_relevance: Literal["none", "low", "medium", "high"]
    rationale: str


@dataclass
class TargetTurn:
    sample: PersonalizedSample
    session: SessionData
    session_file: str
    turn_idx: int
    user_msg: str
    assistant_reply: str
    gold_score: int
    gold_reason: str

    @property
    def sample_id(self) -> str:
        return (
            f"{self.sample.user}__{self.sample.target_task}__"
            f"{self.session_file}__turn_{self.turn_idx}"
        )


def load_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _iter_target_turns(sample: PersonalizedSample) -> list[TargetTurn]:
    turns: list[TargetTurn] = []
    for session in sample.target_sessions:
        session_file = os.path.basename(session.file_path)
        assistant_turn_idx = 0
        last_user_msg = ""
        for utt in session.history:
            role = utt.get("role")
            if role == "user":
                last_user_msg = utt.get("content", "")
                continue
            if role != "assistant":
                continue
            if assistant_turn_idx >= len(session.satisfaction_scores):
                break
            turns.append(
                TargetTurn(
                    sample=sample,
                    session=session,
                    session_file=session_file,
                    turn_idx=assistant_turn_idx,
                    user_msg=last_user_msg,
                    assistant_reply=utt.get("content", ""),
                    gold_score=int(session.satisfaction_scores[assistant_turn_idx]),
                    gold_reason=session.dissatisfaction_reasons[assistant_turn_idx],
                )
            )
            assistant_turn_idx += 1
    return turns


def load_target_turns(
    split: str,
    train_ratio: float,
    seed: int,
    min_history_sessions: int,
    target_tasks: list[str] | None,
) -> dict[str, TargetTurn]:
    turns: dict[str, TargetTurn] = {}
    samples = build_personalized_samples(
        split=split,
        train_ratio=train_ratio,
        seed=seed,
        min_history_sessions=min_history_sessions,
        target_tasks=target_tasks,
    )
    for sample in samples:
        for turn in _iter_target_turns(sample):
            turns[turn.sample_id] = turn
    return turns


def _current_turn_exchange(session: SessionData, turn_idx: int) -> str:
    """Return only the user message and assistant reply for the current turn."""
    assistant_seen = 0
    last_user_msg = ""
    for utt in session.history:
        role = utt.get("role")
        content = str(utt.get("content", "")).strip()
        if role == "user":
            last_user_msg = content
            continue
        if role != "assistant":
            continue
        if assistant_seen == turn_idx:
            return f"用户: {last_user_msg}\n当前助手回复: {content}"
        assistant_seen += 1
    return ""


def build_annotation_prompt(turn: TargetTurn) -> str:
    current_exchange = _current_turn_exchange(turn.session, turn.turn_idx)
    return (
        "请判断当前助手回复是否包含对用户任务有实质帮助的内容，而不是只做澄清、反问、寒暄或流程确认。\n\n"
        "判定标准：\n"
        "- substantive_answer: 当前回复主要给出具体建议、方案、步骤、解释、推荐、清单、判断或可执行内容。\n"
        "- mixed_answer: 当前回复既提出澄清问题，也给出一定实质内容；只要实质内容足以被用户评价，就算 has_substantive_content=true。\n"
        "- clarifying_question: 当前回复主要是在追问需求、确认偏好、询问约束，几乎没有可执行任务内容。\n"
        "- ack_or_meta: 当前回复主要是确认收到、说明将要做什么、道歉、寒暄或元说明。\n"
        "- other: 不属于以上情况。\n\n"
        "注意：\n"
        "- 不要根据 gold_score 或满意度判断；只判断回复内容性质。\n"
        "- 即使回复在第 0 轮，只要已经给出具体方案，也应标为 substantive_answer 或 mixed_answer。\n"
        "- 如果只是问预算、时间、地点、偏好等，且没有实质建议，应标为 clarifying_question。\n\n"
        f"任务类型：{turn.sample.target_task}\n"
        f"任务背景：{turn.session.task_context}\n\n"
        f"当前轮对话：\n{current_exchange}\n\n"
        "请只输出一个 JSON object，不要输出 <think>、Markdown 或解释文字。rationale 控制在 40 个汉字以内。\n"
        "{\n"
        '  "has_substantive_content": true 或 false,\n'
        '  "content_type": "substantive_answer | mixed_answer | clarifying_question | ack_or_meta | other",\n'
        '  "task_relevance": "none | low | medium | high",\n'
        '  "rationale": "简短理由"\n'
        "}"
    )


def _message_content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _extract_json_object(text: str) -> dict | None:
    raw = text.strip()
    if not raw:
        return None

    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    if raw.startswith("<think>") and "</think>" in raw:
        raw = raw.split("</think>", 1)[1].strip()

    candidates: list[str] = []
    if "{" in raw and "}" in raw:
        start = raw.find("{")
        depth = 0
        in_string = False
        escape = False
        for idx, ch in enumerate(raw[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(raw[start: idx + 1])
                    break

        last_start = raw.rfind("{")
        last_end = raw.rfind("}")
        if last_start >= 0 and last_end > last_start:
            candidates.append(raw[last_start: last_end + 1])

    candidates.append(raw)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    before_sleep=before_sleep_log(logger, log_level=40),
)
def call_annotation(
    prompt: str,
    model: str,
    timeout: int = 120,
    max_tokens: int = 256,
    use_schema_parse: bool = False,
) -> TurnContentAnnotation:
    if use_schema_parse:
        response = client.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "/no_think\nYou are a careful dialogue annotation expert. Return only JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format=TurnContentAnnotation,
            timeout=timeout,
        ).choices[0].message
        if response.parsed:
            return response.parsed
        raise RuntimeError(f"Structured annotation parse failed: {response.refusal or 'no content'}")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "/no_think\n"
                    "You are a careful dialogue annotation expert. "
                    "Return only a compact JSON object. Do not output hidden reasoning."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=timeout,
    ).choices[0].message
    raw_text = _message_content_to_text(getattr(response, "content", ""))
    payload = _extract_json_object(raw_text)
    if payload is None:
        preview = raw_text[:300].replace("\n", "\\n")
        raise RuntimeError(f"Raw annotation JSON parse failed; raw_preview={preview}")
    return TurnContentAnnotation.model_validate(payload)


def load_existing_annotations(path: str) -> dict[str, dict]:
    if not path or not os.path.exists(path):
        return {}
    records = load_jsonl(path)
    return {str(r["sample_id"]): r for r in records if r.get("sample_id")}


def command_annotate(args) -> None:
    ids: list[str] | None = None
    if args.result_file:
        ids = [str(r["sample_id"]) for r in load_jsonl(args.result_file)]
        logger.info(f"Loaded {len(ids)} sample ids from {args.result_file}")

    target_tasks = args.target_tasks if args.target_tasks else None
    turn_map = load_target_turns(
        split=args.split,
        train_ratio=args.train_ratio,
        seed=args.split_seed,
        min_history_sessions=args.min_history_sessions,
        target_tasks=target_tasks,
    )
    if ids is None:
        ids = sorted(turn_map)

    if args.limit > 0:
        ids = ids[: args.limit]

    existing = load_existing_annotations(args.output_jsonl)
    output_records = list(existing.values()) if args.resume else []
    skipped = 0

    for sample_id in tqdm(ids, desc="annotating turns"):
        if args.resume and sample_id in existing:
            skipped += 1
            continue
        turn = turn_map.get(sample_id)
        if turn is None:
            logger.warning(f"Missing source turn for sample_id={sample_id}")
            continue
        prompt = build_annotation_prompt(turn)
        ann = call_annotation(
            prompt,
            args.model,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            use_schema_parse=args.use_schema_parse,
        )
        output_records.append(
            {
                "sample_id": sample_id,
                "user": turn.sample.user,
                "target_task": turn.sample.target_task,
                "target_file": turn.session_file,
                "turn_idx": turn.turn_idx,
                "gold_score": turn.gold_score,
                "gold_reason": turn.gold_reason,
                "user_msg": turn.user_msg,
                "assistant_reply": turn.assistant_reply,
                **ann.model_dump(),
            }
        )
        if args.save_every > 0 and len(output_records) % args.save_every == 0:
            save_jsonl(output_records, args.output_jsonl)

    save_jsonl(output_records, args.output_jsonl)
    logger.info(
        f"Saved annotations: {len(output_records)} to {args.output_jsonl} "
        f"(skipped_existing={skipped})"
    )


def _sat(score: int | float) -> int:
    return int(int(score) >= 4)


def _metrics(records: list[dict]) -> dict:
    if not records:
        return {"n": 0}
    gold = [int(r["gold_score"]) for r in records]
    pred = [int(r["pred_score"]) for r in records]
    gold_bin = [_sat(x) for x in gold]
    pred_bin = [_sat(x) for x in pred]
    n_dsat = sum(1 for x in gold_bin if x == 0)
    n_sat = sum(1 for x in gold_bin if x == 1)
    out = {
        "n": len(records),
        "gold_mean": sum(gold) / len(gold),
        "pred_mean": sum(pred) / len(pred),
        "gold_dsat_rate": n_dsat / len(records),
        "pred_dsat_rate": sum(1 for x in pred_bin if x == 0) / len(records),
        "mae": float(mean_absolute_error(gold, pred)),
        "rmse": math.sqrt(sum((g - p) ** 2 for g, p in zip(gold, pred)) / len(records)),
        "bin_acc": float(accuracy_score(gold_bin, pred_bin)),
        "f1_dsat": float(f1_score(gold_bin, pred_bin, pos_label=0, zero_division=0)),
        "precision_dsat": float(precision_score(gold_bin, pred_bin, pos_label=0, zero_division=0)),
        "recall_dsat": float(recall_score(gold_bin, pred_bin, pos_label=0, zero_division=0)),
        "false_sat": (
            sum(1 for g, p in zip(gold_bin, pred_bin) if g == 0 and p == 1) / n_dsat
            if n_dsat else None
        ),
        "false_dsat": (
            sum(1 for g, p in zip(gold_bin, pred_bin) if g == 1 and p == 0) / n_sat
            if n_sat else None
        ),
    }
    out["qwk"] = (
        float(cohen_kappa_score(gold, pred, weights="quadratic", labels=[1, 2, 3, 4, 5]))
        if len(set(pred)) > 1 else None
    )
    out["pearson"] = float(pearsonr(gold, pred)[0]) if len(set(pred)) > 1 else None
    out["spearman"] = float(spearmanr(gold, pred)[0]) if len(set(pred)) > 1 else None
    return out


def _parse_named_files(items: list[str]) -> list[tuple[str, str]]:
    named: list[tuple[str, str]] = []
    for item in items:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            name = os.path.splitext(os.path.basename(item))[0]
            path = item
        named.append((name, path))
    return named


def command_analyze(args) -> None:
    annotations = load_jsonl(args.annotations_jsonl)
    ann_by_id = {str(r["sample_id"]): r for r in annotations}
    logger.info(f"Loaded annotations: {len(ann_by_id)}")

    all_results: dict[str, dict] = {}
    for name, path in _parse_named_files(args.result_files):
        records = load_jsonl(path)
        joined: list[dict] = []
        missing = 0
        for record in records:
            ann = ann_by_id.get(str(record.get("sample_id")))
            if ann is None:
                missing += 1
                continue
            joined.append({**record, **{f"content_{k}": v for k, v in ann.items()}})

        groups: dict[str, list[dict]] = {
            "all_annotated": joined,
            "content_like": [r for r in joined if bool(r.get("content_has_substantive_content"))],
            "non_content": [r for r in joined if not bool(r.get("content_has_substantive_content"))],
        }
        by_type: dict[str, list[dict]] = defaultdict(list)
        by_turn_bucket: dict[str, list[dict]] = defaultdict(list)
        for record in joined:
            by_type[str(record.get("content_content_type", "unknown"))].append(record)
            turn_idx = int(record.get("turn_idx", 0))
            bucket = str(turn_idx) if turn_idx < 5 else "5+"
            by_turn_bucket[bucket].append(record)

        all_results[name] = {
            "path": path,
            "n_joined": len(joined),
            "n_missing_annotations": missing,
            "groups": {group: _metrics(items) for group, items in groups.items()},
            "by_content_type": {group: _metrics(items) for group, items in sorted(by_type.items())},
            "by_turn": {
                group: _metrics(by_turn_bucket[group])
                for group in ["0", "1", "2", "3", "4", "5+"]
                if group in by_turn_bucket
            },
        }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved analysis: {args.output_json}")

    for name, result in all_results.items():
        logger.info(f"\n{name}: joined={result['n_joined']} missing={result['n_missing_annotations']}")
        for group in ("content_like", "non_content"):
            m = result["groups"][group]
            if m["n"] == 0:
                continue
            logger.info(
                f"  {group:<13} n={m['n']} dsat={m['gold_dsat_rate']:.3f} "
                f"MAE={m['mae']:.4f} QWK={m['qwk']} F1-DSAT={m['f1_dsat']:.4f}"
            )


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Annotate and evaluate substantive assistant turns.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ann = sub.add_parser("annotate", help="LLM-annotate whether turns contain substantive content.")
    p_ann.add_argument("--result_file", type=str, default="", help="Annotate only sample_ids in this JSONL.")
    p_ann.add_argument("--output_jsonl", type=str, required=True)
    p_ann.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    p_ann.add_argument("--vllm_base_url", type=str, default="")
    p_ann.add_argument("--vllm_api_key", type=str, default="EMPTY")
    p_ann.add_argument("--split", type=str, choices=["train", "test", "all"], default="test")
    p_ann.add_argument("--train_ratio", type=float, default=0.2)
    p_ann.add_argument("--split_seed", type=int, default=42)
    p_ann.add_argument("--min_history_sessions", type=int, default=1)
    p_ann.add_argument("--target_tasks", type=str, nargs="*", default=[])
    p_ann.add_argument("--limit", type=int, default=0)
    p_ann.add_argument("--resume", action="store_true", default=True)
    p_ann.add_argument("--no_resume", action="store_false", dest="resume")
    p_ann.add_argument("--save_every", type=int, default=50)
    p_ann.add_argument("--timeout", type=int, default=120)
    p_ann.add_argument("--max_tokens", type=int, default=256)
    p_ann.add_argument(
        "--use_schema_parse",
        action="store_true",
        help="Use SDK schema parsing instead of the faster raw-JSON route.",
    )
    p_ann.set_defaults(func=command_annotate)

    p_eval = sub.add_parser("analyze", help="Analyze result files by content annotation groups.")
    p_eval.add_argument("--annotations_jsonl", type=str, required=True)
    p_eval.add_argument("--result_files", type=str, nargs="+", required=True, metavar="NAME=PATH")
    p_eval.add_argument("--output_json", type=str, default="")
    p_eval.set_defaults(func=command_analyze)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    global client
    if getattr(args, "vllm_base_url", ""):
        client = OpenAI(base_url=args.vllm_base_url, api_key=args.vllm_api_key)
        logger.info(f"vLLM mode: base_url={args.vllm_base_url}")
    args.func(args)


if __name__ == "__main__":
    main()
