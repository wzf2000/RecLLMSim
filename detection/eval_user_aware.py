"""
对已有预测结果文件计算用户感知指标（User-Aware Metrics），
对比全局指标 vs. Per-user aggregation vs. Within-user centering。

支持所有含 user / gold_score / pred_score 字段的 JSONL 结果文件。
"""

import json
import math
import os
from argparse import ArgumentParser

from loguru import logger
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    cohen_kappa_score,
    mean_absolute_error,
    root_mean_squared_error,
)

from user_aware_metrics import compute_user_aware_metrics, print_user_aware_metrics


# ──────────────────────────────────────────────────────────────────────────────
# 全局基准指标（不含用户感知，用于对比）
# ──────────────────────────────────────────────────────────────────────────────

def compute_global_metrics(
    gold: list[float],
    pred: list[float],
) -> dict[str, float]:
    import warnings, numpy as np
    g = [float(v) for v in gold]
    p = [float(v) for v in pred]
    g_arr = [round(v) for v in g]
    p_arr = [max(1, min(5, round(v))) for v in p]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pear = pearsonr(g, p)[0]
        spear = spearmanr(g, p)[0]
    try:
        kappa = cohen_kappa_score(g_arr, p_arr, weights="quadratic", labels=list(range(1, 6)))
    except Exception:
        kappa = float("nan")
    return {
        "global_mae":      float(mean_absolute_error(g, p)),
        "global_rmse":     float(root_mean_squared_error(g, p)),
        "global_pearson":  float(pear),
        "global_spearman": float(spear),
        "global_kappa":    float(kappa),
        "n_samples":       len(g),
    }


def print_global_metrics(m: dict, header: str = "") -> None:
    sep = "=" * 62
    if header:
        logger.info(sep)
        logger.info(f"  {header}")
    logger.info(sep)
    logger.info("  ── 全局基准指标（未考虑用户偏差）──────────────────────────")
    logger.info(f"  样本数:  {m['n_samples']}")
    def f(v): return f"{v:.4f}" if not math.isnan(v) else "  N/A  "
    logger.info(f"  MAE:     {f(m['global_mae'])}    RMSE:     {f(m['global_rmse'])}")
    logger.info(f"  Pearson: {f(m['global_pearson'])}    Spearman: {f(m['global_spearman'])}")
    logger.info(f"  Kappa:   {f(m['global_kappa'])}")
    logger.info(sep)


# ──────────────────────────────────────────────────────────────────────────────
# 预设模型配置
# ──────────────────────────────────────────────────────────────────────────────

PRESET_MODELS = {
    "sft": {
        "path": "outputs/evaluation/sft_qwen3_from_self_distill_v3_results.jsonl",
        "name": "SFT (Qwen3, self-distill v3)",
    },
    "grpo": {
        "path": "outputs/evaluation/grpo_from_sft_qwen3_from_gpt5_correct_reasoning_results.jsonl",
        "name": "GRPO (from SFT, GPT-4 correct reasoning)",
    },
    "ordinal": {
        "path": "outputs/evaluation/ordinal_lora_checkpoint1628_results.jsonl",
        "name": "Ordinal-LoRA (checkpoint-1628)",
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# 加载 & 解析
# ──────────────────────────────────────────────────────────────────────────────

def load_records(path: str) -> tuple[list[float], list[float], list[str]]:
    """从 JSONL 加载 gold_score, pred_score, user。"""
    gold, pred, users = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            gs = r.get("gold_score")
            ps = r.get("pred_score")
            u  = r.get("user", "unknown")
            if gs is None or ps is None:
                continue
            gold.append(float(gs))
            pred.append(float(ps))
            users.append(str(u))
    return gold, pred, users


# ──────────────────────────────────────────────────────────────────────────────
# 对比摘要表
# ──────────────────────────────────────────────────────────────────────────────

def print_comparison_table(all_results: dict[str, dict]) -> None:
    """输出全局 vs. per-user vs. within-user 的对比摘要表。"""
    sep = "=" * 90
    logger.info(sep)
    logger.info("  指标对比摘要表")
    logger.info(sep)

    metrics_to_show = [
        ("Pearson",  "global_pearson",  "pu_pearson",  "wc_pearson"),
        ("Spearman", "global_spearman", "pu_spearman", "wc_spearman"),
        ("Kappa",    "global_kappa",    "pu_kappa",    None),
        ("MAE",      "global_mae",      "pu_mae",      "wc_mae"),
        ("RMSE",     "global_rmse",     "pu_rmse",     "wc_rmse"),
    ]

    def f(v):
        return f"{v:.4f}" if v is not None and not math.isnan(v) else " N/A  "

    col_w = 38
    header = (
        f"  {'模型':<{col_w}}"
        f"  {'Global':>8}"
        f"  {'PerUser':>8}"
        f"  {'Centered':>8}"
        f"  {'ΔG→PU':>7}"
        f"  {'ΔG→WC':>7}"
    )
    for metric_name, gk, puk, wck in metrics_to_show:
        logger.info("")
        logger.info(f"  [{metric_name}]")
        logger.info(header)
        logger.info("  " + "-" * 84)
        for model_key, results in all_results.items():
            gm  = results.get("global", {})
            pum = results.get("per_user", {})
            wcm = results.get("within_user", {})
            name = results.get("name", model_key)

            gv   = gm.get(gk)
            puv  = pum.get(puk)
            wcv  = wcm.get(wck) if wck else None

            delta_pu = (puv - gv)  if (puv is not None and gv is not None
                                        and not math.isnan(puv) and not math.isnan(gv)) else None
            delta_wc = (wcv - gv)  if (wcv is not None and gv is not None
                                        and not math.isnan(wcv) and not math.isnan(gv)) else None

            def fd(v): return f"{v:+.4f}" if v is not None else "  N/A "

            row = (
                f"  {name:<{col_w}}"
                f"  {f(gv):>8}"
                f"  {f(puv):>8}"
                f"  {f(wcv):>8}"
                f"  {fd(delta_pu):>7}"
                f"  {fd(delta_wc):>7}"
            )
            logger.info(row)

    logger.info("")
    logger.info("  注：ΔG→PU = per-user - global（正值=用户感知指标更高）")
    logger.info("      ΔG→WC = within-centered - global")
    logger.info(sep)


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser(description="用户感知指标评估：per-user aggregation + within-user centering")
    parser.add_argument(
        "--models", nargs="+",
        default=list(PRESET_MODELS.keys()),
        choices=list(PRESET_MODELS.keys()),
        help="要评估的预设模型",
    )
    parser.add_argument(
        "--result_file", type=str, default="",
        help="额外的自定义 JSONL 结果文件路径",
    )
    parser.add_argument(
        "--min_samples", type=int, default=3,
        help="per-user 相关系数/kappa 计算的最小样本数阈值（默认 3）",
    )
    parser.add_argument(
        "--output_json", type=str, default="",
        help="将结果保存为 JSON（留空则不保存）",
    )
    args = parser.parse_args()

    all_results: dict[str, dict] = {}

    targets: list[tuple[str, str, str]] = []  # (key, path, name)
    for key in args.models:
        cfg = PRESET_MODELS[key]
        path = os.path.join(os.path.dirname(__file__), cfg["path"])
        targets.append((key, path, cfg["name"]))
    if args.result_file:
        key = os.path.splitext(os.path.basename(args.result_file))[0]
        targets.append((key, args.result_file, key))

    for key, path, name in targets:
        if not os.path.exists(path):
            logger.warning(f"文件不存在，跳过: {path}")
            continue

        gold, pred, users = load_records(path)
        if not gold:
            logger.warning(f"无有效记录，跳过: {path}")
            continue

        logger.info(f"\n{'='*62}")
        logger.info(f"  模型: {name}")
        logger.info(f"  文件: {path}")
        logger.info(f"  总样本={len(gold)},  用户数={len(set(users))}")

        # 全局基准
        gm = compute_global_metrics(gold, pred)
        print_global_metrics(gm)

        # 用户感知指标
        ua = compute_user_aware_metrics(gold, pred, users, min_samples=args.min_samples)
        print_user_aware_metrics(ua, header=f"用户感知指标 — {name}")

        all_results[key] = {
            "name":       name,
            "global":     gm,
            "per_user":   ua,
            "within_user": ua,   # pu_ 和 wc_ 均在 ua 中
        }

    # 对比摘要表
    if len(all_results) > 1:
        print_comparison_table(all_results)

    # 保存
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存至: {args.output_json}")


if __name__ == "__main__":
    main()
