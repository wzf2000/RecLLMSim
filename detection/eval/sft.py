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
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from peft import PeftModel
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    root_mean_squared_error,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from trace.collect_api import get_rows_from_split  # noqa: E402
from trace.sft import (  # noqa: E402
    build_source_text,
    load_jsonl,
    parse_model_json,
    resolve_prompt_for_row,
)
from lib.satisfaction_constants import get_reason_to_id  # noqa: E402


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


# ── Generation backends ────────────────────────────────────────────────────────

def _generate_vllm(
    token_ids_list: list[list[int]],
    base_model_name: str,
    lora_checkpoint: str | None,
    max_new_tokens: int,
    max_input_length: int,
    max_lora_rank: int = 64,
    gpu_memory_utilization: float = 0.90,
) -> list[str]:
    """Batch-generate with vLLM. Uses native LoRA support when lora_checkpoint is given."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm_kwargs: dict[str, Any] = dict(
        model=base_model_name,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=max_input_length + max_new_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    lora_req: LoRARequest | None = None
    if lora_checkpoint is not None:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = max_lora_rank
        lora_req = LoRARequest("sft", 1, lora_checkpoint)

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens, skip_special_tokens=True)
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in token_ids_list],
        sampling_params,
        lora_request=lora_req,
    )
    texts = [o.outputs[0].text for o in outputs]
    del llm
    torch.cuda.empty_cache()
    return texts


def _generate_hf(
    token_ids_list: list[list[int]],
    base_model_name: str,
    checkpoint: str | None,
    max_new_tokens: int,
    tokenizer: Any,
) -> list[str]:
    """Sequential generation with HuggingFace transformers (fallback)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, trust_remote_code=True,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    model = PeftModel.from_pretrained(base, checkpoint) if checkpoint is not None else base
    model.to(device).eval()

    texts: list[str] = []
    with torch.inference_mode():
        for token_ids in tqdm(token_ids_list, desc="generate (hf)"):
            enc = {
                "input_ids": torch.tensor([token_ids], device=device),
                "attention_mask": torch.ones(1, len(token_ids), dtype=torch.long, device=device),
            }
            out_ids = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
            gen = out_ids[0][len(token_ids):]
            texts.append(tokenizer.decode(gen, skip_special_tokens=True))
    del model
    torch.cuda.empty_cache()
    return texts


# ── Main evaluation function ───────────────────────────────────────────────────

def evaluate(
    checkpoint: str | None,
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
    think_wrap: str = "qwen3",
    limit: int | None = None,
    backend: str = "vllm",
    max_lora_rank: int = 64,
    gpu_memory_utilization: float = 0.90,
    output_jsonl: str = "",
) -> dict[str, float]:
    # ── Tokenizer (needed for prompt building in both backends) ────────────
    tok_src = checkpoint if checkpoint is not None else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_tag = "base_model" if checkpoint is None else f"sft:{os.path.basename(checkpoint.rstrip('/'))}"

    # ── Data ───────────────────────────────────────────────────────────────
    if test_jsonl:
        rows = load_jsonl(test_jsonl)
        data_source = f"jsonl:{test_jsonl}|{model_tag}"
    else:
        raw_rows = get_rows_from_split(
            split=data_split, split_seed=split_seed,
            train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
        )
        rows = [raw_row_to_eval_record(r) for r in raw_rows]
        data_source = f"split:{data_split}|{model_tag}"
    if limit is not None:
        rows = rows[:limit]

    # ── Build prompts (token_ids) ──────────────────────────────────────────
    source_budget = max(1, max_length - max_new_tokens)
    all_token_ids: list[list[int]] = []
    for row in tqdm(rows, desc="build prompts"):
        prompt = resolve_prompt_for_row(
            row, tokenizer, max_length=max_length,
            include_reasoning_content=include_reasoning_content,
            max_history_turns_cap=max_history_turns,
            source_budget_override=source_budget,
            think_wrap=think_wrap,
        )
        source_text = build_source_text(tokenizer, prompt)
        token_ids = tokenizer(
            source_text, add_special_tokens=False, truncation=True, max_length=max_length,
        )["input_ids"]
        all_token_ids.append(token_ids)

    # ── Generate ───────────────────────────────────────────────────────────
    logger.info(f"Backend={backend}, samples={len(rows)}, model_tag={model_tag}")
    if backend == "vllm":
        texts = _generate_vllm(
            all_token_ids, base_model_name, checkpoint,
            max_new_tokens, max_length, max_lora_rank, gpu_memory_utilization,
        )
    else:
        texts = _generate_hf(all_token_ids, base_model_name, checkpoint, max_new_tokens, tokenizer)

    # ── Parse & metrics ────────────────────────────────────────────────────
    valid_reasons = set(get_reason_to_id().keys())
    y_score: list[int] = []
    y_hat_score: list[int] = []
    y_reason: list[str] = []
    y_hat_reason: list[str] = []
    parse_fail = 0
    sample_records: list[dict[str, Any]] = []

    for text, row in zip(texts, rows):
        ps, pr = parse_model_json(text)
        meta = row.get("meta", {})
        record: dict[str, Any] = {
            "user": meta.get("user", ""),
            "gold_score": int(row["gold_score"]),
            "gold_reason": str(row["gold_reason"]),
            "pred_score": ps,
            "pred_reason_raw": pr,
            "pred_reason": None,
            "raw_text": text,
            "parse_ok": ps is not None,
        }
        if ps is None:
            parse_fail += 1
        else:
            pr_norm = normalize_reason(pr, valid_reasons)
            record["pred_reason"] = pr_norm
            y_score.append(int(row["gold_score"]))
            y_hat_score.append(ps)
            y_reason.append(str(row["gold_reason"]))
            y_hat_reason.append(pr_norm)
        sample_records.append(record)

    if not y_score:
        raise RuntimeError("No valid predictions (all parse failures). Check generation / prompt template.")

    mae = mean_absolute_error(y_score, y_hat_score)
    rmse = root_mean_squared_error(y_score, y_hat_score)
    kappa = cohen_kappa_score(y_score, y_hat_score, weights="quadratic")
    acc_score = accuracy_score(y_score, y_hat_score)
    pearson_r = float(pearsonr(y_score, y_hat_score).statistic)
    spearman_r = float(spearmanr(y_score, y_hat_score).statistic)
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
        "score_pearson_r": pearson_r,
        "score_spearman_r": spearman_r,
        "score_accuracy": float(acc_score),
        "reason_accuracy": float(acc_reason),
        "reason_f1_weighted": float(f1w),
        "reason_f1_macro": float(f1m),
    }

    if output_jsonl:
        os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for rec in sample_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(sample_records)} sample records to {output_jsonl}")

    return metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="", help="SFT 输出目录（含 adapter）；可留空仅评测 base model")
    p.add_argument("--eval_base_model", action="store_true", help="额外评测裸 base model（无 LoRA）以作对比")
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
        help="与训练时 resolve_prompt 预算一致；若训练使用了 --include_reasoning_content，此处应打开并建议增大 --max_new_tokens",
    )
    p.add_argument(
        "--think_wrap",
        type=str,
        default="qwen3",
        choices=["qwen3", "none"],
        help="须与训练 sft_from_traces.py 的 --think_wrap 一致，用于 resolve_prompt 长度估计",
    )
    p.add_argument("--limit", type=int, default=None, help="只测前 N 条，调试用")
    p.add_argument("--metrics_json", type=str, default="", help="可选，将指标写入该 json 文件")
    p.add_argument(
        "--output_jsonl", type=str, default="",
        help="可选，将每条样本的预测结果写入该 jsonl 文件（含 gold/pred score/reason/raw_text/parse_ok）；"
             "同时评测 SFT 与 base 时自动加 _sft/_base 后缀",
    )
    p.add_argument(
        "--backend", type=str, default="vllm", choices=["vllm", "hf"],
        help="推理后端：vllm（默认，批量推理，大幅加速）或 hf（逐条 transformers，兼容性强）",
    )
    p.add_argument("--max_lora_rank", type=int, default=64, help="vLLM LoRA 最大 rank，须 >= 训练时的 lora_r")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90, help="vLLM GPU 显存利用率（0~1）")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.checkpoint and not args.eval_base_model:
        logger.error("错误：--checkpoint 和 --eval_base_model 至少需要指定一个。")
        sys.exit(1)

    eval_kwargs = dict(
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
        think_wrap=args.think_wrap,
        limit=args.limit,
        backend=args.backend,
        max_lora_rank=args.max_lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    # 同时评测两个模型时，自动为 output_jsonl 加 _sft/_base 后缀以避免覆盖
    both = bool(args.checkpoint) and args.eval_base_model
    def _out_path(tag: str) -> str:
        if not args.output_jsonl:
            return ""
        if not both:
            return args.output_jsonl
        p = Path(args.output_jsonl)
        return str(p.parent / f"{p.stem}_{tag}{p.suffix}")

    all_metrics: dict[str, Any] = {}

    if args.checkpoint:
        logger.info("=== SFT Model ===")
        m_sft = evaluate(checkpoint=args.checkpoint, output_jsonl=_out_path("sft"), **eval_kwargs)
        logger.info(json.dumps(m_sft, ensure_ascii=False, indent=2))
        all_metrics["sft"] = m_sft

    if args.eval_base_model:
        logger.info("=== Base Model ===")
        m_base = evaluate(checkpoint=None, output_jsonl=_out_path("base"), **eval_kwargs)
        logger.info(json.dumps(m_base, ensure_ascii=False, indent=2))
        all_metrics["base"] = m_base

    if args.metrics_json:
        os.makedirs(os.path.dirname(args.metrics_json) or ".", exist_ok=True)
        # 若只评测了一种模型，直接写该 dict；否则写包含两个键的 dict
        output = all_metrics if len(all_metrics) > 1 else next(iter(all_metrics.values()))
        with open(args.metrics_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
