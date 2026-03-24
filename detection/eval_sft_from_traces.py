"""
评测 SFT 后的开源模型：对满意度分数与原因生成结果计算与 gold 对齐的指标。

默认测试集与 `collect_api_model_traces.py` 一致：从 `get_satisfaction_data()` 展开样本后，
按用户做 train/val/test 划分（默认 0.8/0.1/0.1, seed=42），使用 **test** 划分。

可选 `--test_jsonl` 覆盖为自定义轨迹文件（调试用途）。

用法（在 detection 目录下）：

python eval_sft_from_traces.py \\
  --checkpoint ./ckpts/sft_from_api_traces \\
  --base_model_name Qwen/Qwen3-8B \\
  --max_length 2048 \\
  --max_new_tokens 512
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    root_mean_squared_error,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from collect_api_model_traces import get_rows_from_split  # noqa: E402
from sft_from_traces import (  # noqa: E402
    build_source_text,
    load_jsonl,
    resolve_prompt_for_row,
)
from satisfaction_constants import get_reason_to_id  # noqa: E402


def raw_row_to_eval_record(row: dict[str, Any]) -> dict[str, Any]:
    """将采集端 preprocess 行转为 eval 用的 record（含 meta，供 resolve_prompt_for_row）。"""
    return {
        "gold_score": int(row["gold_score"]),
        "gold_reason": row["gold_reason"],
        "meta": {
            "user": row["user"],
            "persona": row["persona"],
            "task_context": row["task_context"],
            "history": row["history"],
            "assistant_reply": row["assistant_reply"],
        },
    }


def parse_model_json(text: str) -> tuple[int | None, str | None]:
    """从生成文本中解析 classification 与 reason。"""
    s = text.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    if "{" in s:
        start = s.index("{")
        depth = 0
        end = -1
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > start:
            try:
                obj = json.loads(s[start:end])
                c = int(obj["classification"])
                r = str(obj.get("reason", "")).strip()
                if 1 <= c <= 5:
                    return c, r
            except Exception:
                pass
    for ch in s:
        if ch in "12345":
            score = int(ch)
            return score, None
    return None, None


def normalize_reason(pred: str | None, valid: set[str]) -> str:
    if not pred:
        return "其它" if "其它" in valid else next(iter(valid))
    p = pred.strip()
    if p in valid:
        return p
    for label in valid:
        if label in p:
            return label
    return "其它" if "其它" in valid else next(iter(valid))


@torch.inference_mode()
def evaluate(
    checkpoint: str,
    base_model_name: str,
    test_jsonl: str | None = None,
    data_split: str = "test",
    split_seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_length: int = 2048,
    max_new_tokens: int = 512,
    max_history_turns: int = 5,
    include_reasoning_content: bool = False,
    limit: int | None = None,
) -> dict[str, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    model = PeftModel.from_pretrained(base, checkpoint)
    model.to(device)
    model.eval()

    if test_jsonl:
        rows = load_jsonl(test_jsonl)
        data_source = f"jsonl:{test_jsonl}"
    else:
        raw_rows = get_rows_from_split(
            split=data_split,
            split_seed=split_seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        rows = [raw_row_to_eval_record(r) for r in raw_rows]
        data_source = f"split:{data_split}"

    if limit is not None:
        rows = rows[:limit]

    valid_reasons = set(get_reason_to_id().keys())
    source_budget = max(1, max_length - max_new_tokens)

    y_score: list[int] = []
    y_hat_score: list[int] = []
    y_reason: list[str] = []
    y_hat_reason: list[str] = []
    parse_fail = 0

    for row in tqdm(rows, desc="eval"):
        prompt = resolve_prompt_for_row(
            row,
            tokenizer,
            max_length=max_length,
            include_reasoning_content=include_reasoning_content,
            max_history_turns_cap=max_history_turns,
            source_budget_override=source_budget,
        )
        source_text = build_source_text(tokenizer, prompt)
        enc = tokenizer(
            source_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen = out_ids[0][enc["input_ids"].shape[1]:]
        text = tokenizer.decode(gen, skip_special_tokens=True)

        ps, pr = parse_model_json(text)
        gold_s = int(row["gold_score"])
        gold_r = str(row["gold_reason"])

        if ps is None:
            parse_fail += 1
            continue
        pr_norm = normalize_reason(pr, valid_reasons)

        y_score.append(gold_s)
        y_hat_score.append(ps)
        y_reason.append(gold_r)
        y_hat_reason.append(pr_norm)

    if not y_score:
        raise RuntimeError("No valid predictions (all parse failures). Check generation / prompt template.")

    mae = mean_absolute_error(y_score, y_hat_score)
    rmse = root_mean_squared_error(y_score, y_hat_score)
    kappa = cohen_kappa_score(y_score, y_hat_score, weights="quadratic")
    acc_score = accuracy_score(y_score, y_hat_score)

    acc_reason = accuracy_score(y_reason, y_hat_reason)
    f1w = f1_score(y_reason, y_hat_reason, average="weighted", zero_division=0)
    f1m = f1_score(y_reason, y_hat_reason, average="macro", zero_division=0)

    metrics = {
        "data_source": data_source,
        "n": float(len(rows)),
        "n_evaluated": float(len(y_score)),
        "parse_failures": float(parse_fail),
        "score_mae": float(mae),
        "score_rmse": float(rmse),
        "score_quadratic_kappa": float(kappa),
        "score_accuracy": float(acc_score),
        "reason_accuracy": float(acc_reason),
        "reason_f1_weighted": float(f1w),
        "reason_f1_macro": float(f1m),
    }
    return metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="SFT 输出目录（含 adapter）")
    p.add_argument(
        "--base_model_name",
        type=str,
        required=True,
        help="训练时使用的基座模型名（与保存 adapter 时一致）",
    )
    p.add_argument(
        "--test_jsonl",
        type=str,
        default="",
        help="若指定则从该轨迹 jsonl 评测；默认不填则使用与 collect_api_model_traces 一致的原始数据划分",
    )
    p.add_argument(
        "--data_split",
        type=str,
        default="test",
        choices=["train", "valid", "val", "test", "all"],
        help="未指定 test_jsonl 时使用的数据划分（默认 test，与采集脚本 test 集一致）",
    )
    p.add_argument("--split_seed", type=int, default=42, help="划分随机种子，与采集脚本默认一致")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--max_history_turns", type=int, default=5)
    p.add_argument(
        "--include_reasoning_content",
        action="store_true",
        help="与训练时 resolve_prompt 预算一致（一般推理评测可关）",
    )
    p.add_argument("--limit", type=int, default=None, help="只测前 N 条，调试用")
    p.add_argument("--metrics_json", type=str, default="", help="可选，将指标写入该 json 文件")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    m = evaluate(
        checkpoint=args.checkpoint,
        base_model_name=args.base_model_name,
        test_jsonl=args.test_jsonl or None,
        data_split=args.data_split,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        max_history_turns=args.max_history_turns,
        include_reasoning_content=args.include_reasoning_content,
        limit=args.limit,
    )
    logger.info(json.dumps(m, ensure_ascii=False, indent=2))
    if args.metrics_json:
        os.makedirs(os.path.dirname(args.metrics_json) or ".", exist_ok=True)
        with open(args.metrics_json, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)
