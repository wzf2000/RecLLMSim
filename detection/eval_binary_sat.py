"""
将现有模型的预测结果转换为二分类 SAT/DSAT（>=4 为 SAT，<=3 为 DSAT），
并计算与 SPUR 保持一致的评估指标。

支持三类结果文件：
  - LLM predictor（SFT / GRPO）：含 gold_score, pred_score, parse_ok
  - Ordinal-LoRA：含 gold_score, pred_score, ordinal_probs
"""

import json
import math
import os
from argparse import ArgumentParser

from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from user_aware_metrics import (
    compute_user_aware_binary_metrics,
    print_user_aware_binary_metrics,
)

SAT_LABEL = 1   # score >= 4
DSAT_LABEL = 0  # score <= 3


# ──────────────────────────────────────────────────────────────────────────────
# 加载结果文件
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_binary(score: int | float | None) -> int:
    if score is None:
        return DSAT_LABEL  # 解析失败时保守地预测 DSAT
    return SAT_LABEL if int(score) >= 4 else DSAT_LABEL


def get_sat_confidence(record: dict, source: str) -> float:
    """
    计算 SAT 置信度（用于 AUC）：
      - ordinal：直接使用 ordinal_probs[2]，即 P(score >= 4) 的 sigmoid logit
      - llm（SFT/GRPO）：用 pred_score 归一化到 [0,1]，parse 失败时降为 0.5
    """
    if source == "ordinal":
        probs = record.get("ordinal_probs", [])
        # ordinal_probs[i] = P(score >= i+2)；索引 2 对应 P(score >= 4)
        if len(probs) >= 3:
            return float(probs[2])
        return 0.5
    else:
        # LLM predictor：parse 失败时置信度中性化
        if not record.get("parse_ok", True):
            return 0.5
        pred = int(record.get("pred_score", 3))
        # 1->0.1, 2->0.3, 3->0.5, 4->0.7, 5->0.9
        return (pred - 1) / 4 * 0.8 + 0.1


# ──────────────────────────────────────────────────────────────────────────────
# 指标计算
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    records: list[dict],
    source: str,
    name: str = "",
) -> dict[str, float]:
    gold = [to_binary(r["gold_score"]) for r in records]
    pred = [to_binary(r["pred_score"]) for r in records]
    conf = [get_sat_confidence(r, source) for r in records]

    # parse 成功率（LLM predictor 专属）
    if source != "ordinal":
        parse_rate = sum(r.get("parse_ok", True) for r in records) / len(records)
    else:
        parse_rate = 1.0

    try:
        auc = roc_auc_score(gold, conf)
    except Exception:
        auc = float("nan")

    metrics = {
        "name": name,
        "n_samples": len(records),
        "n_sat_gold": int(sum(gold)),
        "n_dsat_gold": int(len(gold) - sum(gold)),
        "parse_rate": parse_rate,
        "accuracy": accuracy_score(gold, pred),
        "f1_macro": f1_score(gold, pred, average="macro", zero_division=0),
        "f1_sat": f1_score(gold, pred, pos_label=1, average="binary", zero_division=0),
        "f1_dsat": f1_score(gold, pred, pos_label=0, average="binary", zero_division=0),
        "precision_sat": precision_score(gold, pred, pos_label=1, average="binary", zero_division=0),
        "recall_sat": recall_score(gold, pred, pos_label=1, average="binary", zero_division=0),
        "kappa": cohen_kappa_score(gold, pred),
        "auc": auc,
    }
    return metrics


def print_metrics(m: dict, header: str = ""):
    sep = "=" * 60
    logger.info(sep)
    if header:
        logger.info(f"  {header}")
        logger.info(sep)
    logger.info(f"  样本数:       {m['n_samples']}  (SAT={m['n_sat_gold']}, DSAT={m['n_dsat_gold']})")
    if m["parse_rate"] < 1.0:
        logger.info(f"  解析成功率:   {m['parse_rate']:.1%}")
    logger.info(f"  Accuracy:     {m['accuracy']:.4f}")
    logger.info(f"  F1-macro:     {m['f1_macro']:.4f}")
    logger.info(f"  F1-SAT:       {m['f1_sat']:.4f}  (P={m['precision_sat']:.4f}, R={m['recall_sat']:.4f})")
    logger.info(f"  F1-DSAT:      {m['f1_dsat']:.4f}")
    logger.info(f"  Kappa:        {m['kappa']:.4f}")
    auc_str = f"{m['auc']:.4f}" if not math.isnan(m["auc"]) else "N/A"
    logger.info(f"  AUC:          {auc_str}")
    logger.info(sep)


# ──────────────────────────────────────────────────────────────────────────────
# 用户感知指标（二分类）辅助
# ──────────────────────────────────────────────────────────────────────────────

def _compute_ua_from_records(
    records: list[dict],
    source: str,
    min_samples: int = 3,
) -> dict[str, float]:
    """从原始记录中提取二分类标签 / 置信度 / 用户，计算用户感知指标。"""
    gold_bin = [to_binary(r["gold_score"]) for r in records]
    pred_bin = [to_binary(r["pred_score"]) for r in records]
    conf = [get_sat_confidence(r, source) for r in records]
    users = [str(r.get("user", "unknown")) for r in records]
    return compute_user_aware_binary_metrics(
        gold_bin, pred_bin, users, conf, min_samples,
    )


def print_binary_ua_comparison(all_metrics: dict[str, dict]) -> None:
    """输出全局 vs. per-user（用户感知）二分类对比摘要表。"""
    sep = "=" * 90
    logger.info(sep)
    logger.info("  二分类指标对比：全局 vs. Per-user（用户感知）")
    logger.info(sep)

    metrics_to_show = [
        ("Accuracy", "accuracy",  "pu_bin_accuracy"),
        ("F1-macro", "f1_macro",  "pu_bin_f1_macro"),
        ("F1-SAT",   "f1_sat",   "pu_bin_f1_sat"),
        ("F1-DSAT",  "f1_dsat",  "pu_bin_f1_dsat"),
        ("Kappa",    "kappa",    "pu_bin_kappa"),
        ("AUC",      "auc",      "pu_bin_auc"),
    ]

    def f(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return " N/A  "
        return f"{v:.4f}"

    col_w = 38
    header = (
        f"  {'模型':<{col_w}}"
        f"  {'Global':>8}"
        f"  {'PerUser':>8}"
        f"  {'Δ':>7}"
    )
    for metric_name, gk, puk in metrics_to_show:
        logger.info("")
        logger.info(f"  [{metric_name}]")
        logger.info(header)
        logger.info("  " + "-" * 66)
        for _key, data in all_metrics.items():
            name = data.get("name", _key)
            gv  = data.get(gk)
            ua  = data.get("user_aware", {})
            puv = ua.get(puk)

            delta = None
            if (puv is not None and gv is not None
                    and not math.isnan(puv) and not math.isnan(gv)):
                delta = puv - gv

            def fd(v):
                return f"{v:+.4f}" if v is not None else "  N/A "

            row = (
                f"  {name:<{col_w}}"
                f"  {f(gv):>8}"
                f"  {f(puv):>8}"
                f"  {fd(delta):>7}"
            )
            logger.info(row)

    logger.info("")
    logger.info("  注：Δ = PerUser - Global（负值表示全局指标因用户间差异而虚高）")
    logger.info(sep)


# ──────────────────────────────────────────────────────────────────────────────
# 预设模型配置
# ──────────────────────────────────────────────────────────────────────────────

PRESET_MODELS = {
    "sft": {
        "path": "outputs/evaluation/sft_qwen3_from_self_distill_v3_results.jsonl",
        "source": "llm",
        "name": "SFT (Qwen3, self-distill v3)",
    },
    "grpo": {
        "path": "outputs/evaluation/grpo_from_sft_qwen3_from_gpt5_correct_reasoning_results.jsonl",
        "source": "llm",
        "name": "GRPO (from SFT, GPT-4 correct reasoning)",
    },
    "ordinal": {
        "path": "outputs/evaluation/ordinal_lora_checkpoint1628_results.jsonl",
        "source": "ordinal",
        "name": "Ordinal-LoRA (checkpoint-1628)",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser(
        description="将已有预测结果转为二分类 SAT/DSAT 并计算与 SPUR 一致的指标"
    )
    parser.add_argument(
        "--models", nargs="+",
        default=list(PRESET_MODELS.keys()),
        choices=list(PRESET_MODELS.keys()),
        help="要评估的模型（默认全部）",
    )
    parser.add_argument(
        "--result_file", type=str, default="",
        help="自定义结果文件路径（配合 --source 使用）",
    )
    parser.add_argument(
        "--source", type=str, default="llm", choices=["llm", "ordinal"],
        help="自定义文件类型：llm（SFT/GRPO）或 ordinal",
    )
    parser.add_argument(
        "--min_samples", type=int, default=3,
        help="per-user 指标计算的最小样本数阈值（默认 3）",
    )
    parser.add_argument(
        "--output_json", type=str, default="",
        help="将所有模型指标保存为 JSON（留空则不保存）",
    )
    args = parser.parse_args()

    all_metrics: dict[str, dict] = {}

    # 自定义文件
    if args.result_file:
        records = load_jsonl(args.result_file)
        m = compute_metrics(records, source=args.source, name=os.path.basename(args.result_file))
        print_metrics(m, header=m["name"])
        ua = _compute_ua_from_records(records, args.source, args.min_samples)
        print_user_aware_binary_metrics(ua, header=f"用户感知（二分类）— {m['name']}")
        m["user_aware"] = ua
        all_metrics[m["name"]] = m

    # 预设模型
    for key in args.models:
        cfg = PRESET_MODELS[key]
        path = os.path.join(os.path.dirname(__file__), cfg["path"])
        if not os.path.exists(path):
            logger.warning(f"结果文件不存在，跳过: {path}")
            continue
        records = load_jsonl(path)
        m = compute_metrics(records, source=cfg["source"], name=cfg["name"])
        print_metrics(m, header=m["name"])
        ua = _compute_ua_from_records(records, cfg["source"], args.min_samples)
        print_user_aware_binary_metrics(ua, header=f"用户感知（二分类）— {cfg['name']}")
        m["user_aware"] = ua
        all_metrics[key] = m

    # 对比摘要表（全局）
    if len(all_metrics) > 1:
        sep = "=" * 60
        logger.info(sep)
        logger.info("  对比摘要（二分类 SAT/DSAT — 全局）")
        logger.info(sep)
        header = f"  {'模型':<38}  {'Acc':>6}  {'F1-mac':>6}  {'F1-SAT':>6}  {'F1-DSA':>6}  {'Kappa':>6}  {'AUC':>6}"
        logger.info(header)
        logger.info("  " + "-" * 56)
        for key, m in all_metrics.items():
            auc_s = f"{m['auc']:.4f}" if not math.isnan(m["auc"]) else "  N/A "
            row = (
                f"  {m['name']:<38}"
                f"  {m['accuracy']:>6.4f}"
                f"  {m['f1_macro']:>6.4f}"
                f"  {m['f1_sat']:>6.4f}"
                f"  {m['f1_dsat']:>6.4f}"
                f"  {m['kappa']:>6.4f}"
                f"  {auc_s:>6}"
            )
            logger.info(row)
        logger.info(sep)

        # 对比摘要表（全局 vs. 用户感知）
        print_binary_ua_comparison(all_metrics)

    # 保存
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"指标已保存至: {args.output_json}")


if __name__ == "__main__":
    main()
