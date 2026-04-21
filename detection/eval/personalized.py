"""
个性化满意度预测评估脚本

在标准指标（MAE / RMSE / Pearson / Spearman / Kappa）基础上，
额外计算个性化增益（Personalization Gain）指标：

  Personalization Gain (PG) = MAE_baseline - MAE_personalized
    > 0 表示个性化版本优于无记忆 baseline
    < 0 表示个性化带来了负面影响

支持功能：
  1. 单文件评估（输出完整指标）
  2. 多文件对比（对比表：有记忆 vs. 无记忆，不同 update mode）
  3. 按任务类型 / 历史 session 数分层分析
  4. 用户感知指标（per-user aggregation + within-user centering）

输出格式
  文本日志（loguru）+ 可选 JSON 结果文件

运行方式（从 detection/ 目录）：
  # 单文件
  python eval/personalized.py \\
    --result_file outputs/personalized/gpt4o_test_per_session.jsonl

  # 多文件对比
  python eval/personalized.py \\
    --result_files with_memory=outputs/personalized/gpt4o_test_per_session.jsonl \\
                   no_memory=outputs/personalized/gpt4o_test_none.jsonl \\
    --output_json outputs/personalized/comparison.json
"""

from __future__ import annotations

import json
import math
import os
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
from loguru import logger
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)

from lib.user_aware_metrics import (
    compute_user_aware_binary_metrics,
    compute_user_aware_metrics,
    print_user_aware_binary_metrics,
    print_user_aware_metrics,
)


# ──────────────────────────────────────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────────────────────────────────────

def load_records(path: str) -> list[dict]:
    """从 JSONL 加载预测结果，仅保留含 gold_score 和 pred_score 的记录。"""
    records: list[dict] = []
    if not os.path.exists(path):
        logger.warning(f"文件不存在: {path}")
        return records
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("gold_score") is not None and r.get("pred_score") is not None:
                    records.append(r)
            except json.JSONDecodeError:
                continue
    return records


# ──────────────────────────────────────────────────────────────────────────────
# 全局指标
# ──────────────────────────────────────────────────────────────────────────────

def compute_global_metrics(
    gold: list[float],
    pred: list[float],
) -> dict[str, float]:
    import warnings

    g = [float(v) for v in gold]
    p = [float(v) for v in pred]
    g_int = [round(v) for v in g]
    p_int = [max(1, min(5, round(v))) for v in p]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pear = float(pearsonr(g, p)[0])
        spear = float(spearmanr(g, p)[0])
    try:
        kappa = float(
            cohen_kappa_score(g_int, p_int, weights="quadratic", labels=list(range(1, 6)))
        )
    except Exception:
        kappa = float("nan")

    return {
        "mae":      float(mean_absolute_error(g, p)),
        "rmse":     float(root_mean_squared_error(g, p)),
        "pearson":  pear,
        "spearman": spear,
        "kappa":    kappa,
        "n_samples": len(g),
    }


def _fmt(v: float, fmt: str = ".4f") -> str:
    return f"{v:{fmt}}" if not (v != v) else "  N/A  "  # isnan via v!=v


def print_global_metrics(m: dict, label: str = "") -> None:
    sep = "=" * 60
    if label:
        logger.info(sep)
        logger.info(f"  {label}")
    logger.info(sep)
    logger.info(f"  样本数:  {m['n_samples']}")
    logger.info(f"  MAE:     {_fmt(m['mae'])}    RMSE:     {_fmt(m['rmse'])}")
    logger.info(f"  Pearson: {_fmt(m['pearson'])}    Spearman: {_fmt(m['spearman'])}")
    logger.info(f"  Kappa:   {_fmt(m['kappa'])}")
    logger.info(sep)


# ──────────────────────────────────────────────────────────────────────────────
# 3/4 边界（二分类 SAT/DSAT）指标
# ──────────────────────────────────────────────────────────────────────────────

SAT_LABEL = 1   # score >= 4
DSAT_LABEL = 0  # score <= 3


def to_binary_sat(score: int | float | None) -> int:
    if score is None:
        return DSAT_LABEL
    return SAT_LABEL if int(score) >= 4 else DSAT_LABEL


def get_sat_confidence(record: dict) -> float:
    """
    由 1-5 的 pred_score 构造一个 SAT 置信度（用于 AUC / user-aware binary）。
    采用与 eval/binary_sat.py 相同的简单单调映射：
      1->0.1, 2->0.3, 3->0.5, 4->0.7, 5->0.9
    """
    pred = int(record.get("pred_score", 3))
    return (pred - 1) / 4 * 0.8 + 0.1


def compute_boundary_metrics(records: list[dict]) -> dict[str, float]:
    gold = [to_binary_sat(r["gold_score"]) for r in records]
    pred = [to_binary_sat(r["pred_score"]) for r in records]
    conf = [get_sat_confidence(r) for r in records]

    try:
        auc = float(roc_auc_score(gold, conf))
    except Exception:
        auc = float("nan")

    n_sat_gold = int(sum(gold))
    n_dsat_gold = int(len(gold) - n_sat_gold)
    n_sat_pred = int(sum(pred))
    n_dsat_pred = int(len(pred) - n_sat_pred)

    false_sat = 0
    false_dsat = 0
    for g, p in zip(gold, pred):
        if g == DSAT_LABEL and p == SAT_LABEL:
            false_sat += 1
        elif g == SAT_LABEL and p == DSAT_LABEL:
            false_dsat += 1

    return {
        "n_samples": len(records),
        "n_sat_gold": n_sat_gold,
        "n_dsat_gold": n_dsat_gold,
        "n_sat_pred": n_sat_pred,
        "n_dsat_pred": n_dsat_pred,
        "accuracy": float(accuracy_score(gold, pred)),
        "f1_macro": float(f1_score(gold, pred, average="macro", zero_division=0)),
        "f1_sat": float(f1_score(gold, pred, pos_label=1, average="binary", zero_division=0)),
        "f1_dsat": float(f1_score(gold, pred, pos_label=0, average="binary", zero_division=0)),
        "precision_sat": float(precision_score(gold, pred, pos_label=1, average="binary", zero_division=0)),
        "recall_sat": float(recall_score(gold, pred, pos_label=1, average="binary", zero_division=0)),
        "precision_dsat": float(precision_score(gold, pred, pos_label=0, average="binary", zero_division=0)),
        "recall_dsat": float(recall_score(gold, pred, pos_label=0, average="binary", zero_division=0)),
        "kappa": float(cohen_kappa_score(gold, pred)),
        "auc": auc,
        "false_sat_rate": float(false_sat / n_dsat_gold) if n_dsat_gold else float("nan"),
        "false_dsat_rate": float(false_dsat / n_sat_gold) if n_sat_gold else float("nan"),
    }


def print_boundary_metrics(m: dict, label: str = "") -> None:
    sep = "=" * 60
    if label:
        logger.info(sep)
        logger.info(f"  {label}")
    logger.info(sep)
    logger.info(
        f"  样本数: {m['n_samples']}  "
        f"(gold SAT={m['n_sat_gold']}, DSAT={m['n_dsat_gold']}; "
        f"pred SAT={m['n_sat_pred']}, DSAT={m['n_dsat_pred']})"
    )
    logger.info(f"  Accuracy:  {_fmt(m['accuracy'])}    F1-macro: {_fmt(m['f1_macro'])}")
    logger.info(
        f"  F1-SAT:    {_fmt(m['f1_sat'])}    "
        f"Prec-SAT: {_fmt(m['precision_sat'])}    Rec-SAT: {_fmt(m['recall_sat'])}"
    )
    logger.info(
        f"  F1-DSAT:   {_fmt(m['f1_dsat'])}    "
        f"Prec-DSAT: {_fmt(m['precision_dsat'])}    Rec-DSAT: {_fmt(m['recall_dsat'])}"
    )
    logger.info(
        f"  Kappa:     {_fmt(m['kappa'])}    AUC: {_fmt(m['auc'])}"
    )
    logger.info(
        f"  False SAT（把不满意判成满意）: {_fmt(m['false_sat_rate'])}"
        f"    False DSAT（把满意判成不满意）: {_fmt(m['false_dsat_rate'])}"
    )
    logger.info(sep)


# ──────────────────────────────────────────────────────────────────────────────
# 分层分析
# ──────────────────────────────────────────────────────────────────────────────

def stratified_analysis(records: list[dict]) -> dict[str, dict]:
    """
    分层计算 MAE，维度包括：
      - 目标任务类型（target_task）
      - 记忆更新模式（memory_update_mode）
      - 是否使用记忆（with_memory）
    """
    results: dict[str, dict] = {}

    def _group_and_compute(key_fn, label_prefix: str) -> None:
        groups: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
        for r in records:
            k = key_fn(r)
            groups[k][0].append(float(r["gold_score"]))
            groups[k][1].append(float(r["pred_score"]))
        for group_key, (g, p) in sorted(groups.items()):
            full_key = f"{label_prefix}/{group_key}"
            results[full_key] = compute_global_metrics(g, p)

    _group_and_compute(lambda r: r.get("target_task", "unknown"), "by_task")
    _group_and_compute(
        lambda r: str(r.get("memory_update_mode", "unknown")), "by_update_mode"
    )
    _group_and_compute(
        lambda r: "with_memory" if r.get("with_memory") else "no_memory", "by_memory"
    )

    return results


def print_stratified_analysis(results: dict[str, dict]) -> None:
    logger.info("=" * 60)
    logger.info("  分层分析（MAE）")
    logger.info("=" * 60)
    for key, m in sorted(results.items()):
        logger.info(f"  {key:<42}  MAE={_fmt(m['mae'])}  n={m['n_samples']}")
    logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# 分层边界分析
# ──────────────────────────────────────────────────────────────────────────────

def boundary_stratified_analysis(records: list[dict]) -> dict[str, dict]:
    results: dict[str, dict] = {}

    def _group_and_compute(key_fn, label_prefix: str) -> None:
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            groups[key_fn(r)].append(r)
        for group_key, group_records in sorted(groups.items()):
            full_key = f"{label_prefix}/{group_key}"
            results[full_key] = compute_boundary_metrics(group_records)

    _group_and_compute(lambda r: r.get("target_task", "unknown"), "by_task")
    _group_and_compute(
        lambda r: str(r.get("memory_update_mode", "unknown")), "by_update_mode"
    )
    _group_and_compute(
        lambda r: "with_memory" if r.get("with_memory") else "no_memory", "by_memory"
    )

    return results


def print_boundary_stratified_analysis(results: dict[str, dict]) -> None:
    logger.info("=" * 72)
    logger.info("  分层分析（3/4 边界 Accuracy / F1-DSAT）")
    logger.info("=" * 72)
    for key, m in sorted(results.items()):
        logger.info(
            f"  {key:<42}  Acc={_fmt(m['accuracy'])}  "
            f"F1-DSAT={_fmt(m['f1_dsat'])}  n={m['n_samples']}"
        )
    logger.info("=" * 72)


# ──────────────────────────────────────────────────────────────────────────────
# 个性化增益（Personalization Gain）
# ──────────────────────────────────────────────────────────────────────────────

def compute_personalization_gain(
    mem_records: list[dict],
    baseline_records: list[dict],
) -> dict[str, float]:
    """
    计算有记忆 vs. 无记忆 baseline 的个性化增益。

    要求两批记录的 sample_id 可对齐（按 sample_id 匹配）；
    若 sample_id 不存在则按顺序对齐（要求等长）。
    """
    # 尝试用 sample_id 对齐
    baseline_by_id = {r.get("sample_id", i): r for i, r in enumerate(baseline_records)}
    matched_mem, matched_base = [], []
    unmatched = 0
    for r in mem_records:
        sid = r.get("sample_id")
        if sid and sid in baseline_by_id:
            matched_mem.append(r)
            matched_base.append(baseline_by_id[sid])
        else:
            unmatched += 1

    if unmatched > 0:
        logger.warning(f"个性化增益计算：{unmatched} 条记录无法与 baseline 对齐，已跳过。")

    if not matched_mem:
        return {"pg_mae": float("nan"), "pg_rmse": float("nan"), "n_matched": 0}

    gold = [float(r["gold_score"]) for r in matched_mem]
    pred_mem = [float(r["pred_score"]) for r in matched_mem]
    pred_base = [float(r["pred_score"]) for r in matched_base]

    mae_mem = float(mean_absolute_error(gold, pred_mem))
    mae_base = float(mean_absolute_error(gold, pred_base))
    rmse_mem = float(root_mean_squared_error(gold, pred_mem))
    rmse_base = float(root_mean_squared_error(gold, pred_base))

    return {
        "pg_mae":  mae_base - mae_mem,       # > 0 = memory helps
        "pg_rmse": rmse_base - rmse_mem,
        "mae_with_memory": mae_mem,
        "mae_baseline":    mae_base,
        "rmse_with_memory": rmse_mem,
        "rmse_baseline":    rmse_base,
        "n_matched": len(matched_mem),
    }


def per_user_personalization_gain(
    mem_records: list[dict],
    baseline_records: list[dict],
) -> dict[str, float]:
    """
    按用户分别计算 PG，返回加权平均 PG。
    用于分析哪类用户从个性化中受益更多。
    """
    baseline_by_id = {r.get("sample_id", i): r for i, r in enumerate(baseline_records)}

    user_pairs: dict[str, tuple[list, list, list]] = defaultdict(lambda: ([], [], []))
    for r in mem_records:
        sid = r.get("sample_id")
        if sid and sid in baseline_by_id:
            base_r = baseline_by_id[sid]
            user = r.get("user", "unknown")
            user_pairs[user][0].append(float(r["gold_score"]))
            user_pairs[user][1].append(float(r["pred_score"]))
            user_pairs[user][2].append(float(base_r["pred_score"]))

    pg_per_user: list[float] = []
    ns: list[int] = []
    positive_users = 0
    for user, (gold, pred_m, pred_b) in user_pairs.items():
        pg = mean_absolute_error(gold, pred_b) - mean_absolute_error(gold, pred_m)
        pg_per_user.append(pg)
        ns.append(len(gold))
        if pg > 0:
            positive_users += 1

    if not pg_per_user:
        return {}

    return {
        "pu_pg_mae":          float(np.average(pg_per_user, weights=ns)),
        "pu_pg_mae_unweighted": float(np.mean(pg_per_user)),
        "n_users":            len(pg_per_user),
        "n_users_positive_pg": positive_users,
        "pct_users_positive_pg": positive_users / len(pg_per_user),
    }


def print_personalization_gain(pg: dict, pu_pg: dict | None = None) -> None:
    sep = "=" * 60
    logger.info(sep)
    logger.info("  个性化增益（Personalization Gain）")
    logger.info(sep)
    n = pg.get("n_matched", 0)
    logger.info(f"  对齐样本数: {n}")
    pg_mae = pg.get("pg_mae", float("nan"))
    pg_rmse = pg.get("pg_rmse", float("nan"))
    indicator = "↑ memory helps" if pg_mae > 0 else ("↓ memory hurts" if pg_mae < 0 else "= neutral")
    logger.info(
        f"  PG (MAE):  {_fmt(pg_mae)}  {indicator}"
        f"    PG (RMSE): {_fmt(pg_rmse)}"
    )
    logger.info(
        f"  MAE  with_memory={_fmt(pg.get('mae_with_memory', float('nan')))}"
        f"  baseline={_fmt(pg.get('mae_baseline', float('nan')))}"
    )
    logger.info(
        f"  RMSE with_memory={_fmt(pg.get('rmse_with_memory', float('nan')))}"
        f"  baseline={_fmt(pg.get('rmse_baseline', float('nan')))}"
    )
    if pu_pg:
        logger.info("  ── Per-user PG ─────────────────────────────────")
        logger.info(
            f"  加权平均 PG (MAE): {_fmt(pu_pg.get('pu_pg_mae', float('nan')))}"
            f"    未加权: {_fmt(pu_pg.get('pu_pg_mae_unweighted', float('nan')))}"
        )
        logger.info(
            f"  用户总数: {pu_pg.get('n_users', 0)}"
            f"    PG>0 用户数: {pu_pg.get('n_users_positive_pg', 0)}"
            f"  ({pu_pg.get('pct_users_positive_pg', 0):.1%})"
        )
    logger.info(sep)


# ──────────────────────────────────────────────────────────────────────────────
# 多文件对比表
# ──────────────────────────────────────────────────────────────────────────────

def print_comparison_table(all_results: dict[str, dict]) -> None:
    sep = "=" * 80
    metrics = [
        ("MAE",      "mae"),
        ("RMSE",     "rmse"),
        ("Pearson",  "pearson"),
        ("Spearman", "spearman"),
        ("Kappa",    "kappa"),
    ]
    col_w = 40

    logger.info(sep)
    logger.info("  指标对比摘要（全局）")
    logger.info(sep)

    header = f"  {'模型/配置':<{col_w}}" + "".join(f"  {m[0]:>9}" for m in metrics)
    logger.info(header)
    logger.info("  " + "-" * 76)

    for name, r in all_results.items():
        gm = r.get("global", {})
        row = f"  {name:<{col_w}}" + "".join(
            f"  {_fmt(gm.get(mk, float('nan'))):>9}" for _, mk in metrics
        )
        logger.info(row)
    logger.info(sep)


def print_boundary_comparison_table(all_results: dict[str, dict]) -> None:
    sep = "=" * 98
    metrics = [
        ("Acc", "accuracy"),
        ("F1-mac", "f1_macro"),
        ("F1-SAT", "f1_sat"),
        ("F1-DSAT", "f1_dsat"),
        ("Kappa", "kappa"),
        ("AUC", "auc"),
        ("FalseSAT", "false_sat_rate"),
        ("FalseDSAT", "false_dsat_rate"),
    ]
    col_w = 28

    logger.info(sep)
    logger.info("  指标对比摘要（3/4 边界：SAT/DSAT）")
    logger.info(sep)
    header = f"  {'模型/配置':<{col_w}}" + "".join(f"  {m[0]:>8}" for m in metrics)
    logger.info(header)
    logger.info("  " + "-" * 94)

    for name, r in all_results.items():
        bm = r.get("boundary", {})
        row = f"  {name:<{col_w}}" + "".join(
            f"  {_fmt(bm.get(mk, float('nan'))):>8}" for _, mk in metrics
        )
        logger.info(row)
    logger.info(sep)


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_single(
    path: str,
    name: str,
    min_samples: int = 3,
) -> dict:
    """评估单个结果文件，返回完整指标字典。"""
    records = load_records(path)
    if not records:
        logger.warning(f"无有效记录: {path}")
        return {}

    gold = [float(r["gold_score"]) for r in records]
    pred = [float(r["pred_score"]) for r in records]
    users = [str(r.get("user", "unknown")) for r in records]

    logger.info(f"\n{'='*60}")
    logger.info(f"  文件: {path}  ({name})")
    logger.info(f"  总样本: {len(gold)},  用户数: {len(set(users))}")

    gm = compute_global_metrics(gold, pred)
    print_global_metrics(gm, label="全局指标")

    bm = compute_boundary_metrics(records)
    print_boundary_metrics(bm, label="3/4 边界（二分类：SAT/DSAT）")

    ua = compute_user_aware_metrics(gold, pred, users, min_samples=min_samples)
    print_user_aware_metrics(ua, header=f"用户感知指标 — {name}")

    gold_bin = [to_binary_sat(r["gold_score"]) for r in records]
    pred_bin = [to_binary_sat(r["pred_score"]) for r in records]
    conf = [get_sat_confidence(r) for r in records]
    ua_bin = compute_user_aware_binary_metrics(
        gold_bin, pred_bin, users, conf, min_samples=min_samples,
    )
    print_user_aware_binary_metrics(ua_bin, header=f"用户感知（二分类）— {name}")

    strat = stratified_analysis(records)
    print_stratified_analysis(strat)

    bstrat = boundary_stratified_analysis(records)
    print_boundary_stratified_analysis(bstrat)

    return {
        "global": gm,
        "boundary": bm,
        "user_aware": ua,
        "user_aware_boundary": ua_bin,
        "stratified": strat,
        "boundary_stratified": bstrat,
        "n": len(gold),
    }


def main() -> None:
    parser = ArgumentParser(description="个性化满意度预测评估")
    parser.add_argument(
        "--result_file",
        type=str,
        default="",
        help="单个结果文件路径（JSONL）",
    )
    parser.add_argument(
        "--result_files",
        type=str,
        nargs="+",
        default=[],
        metavar="NAME=PATH",
        help=(
            "多个结果文件，格式 name=path，用于对比。\n"
            "例: with_memory=out/mem.jsonl no_memory=out/base.jsonl"
        ),
    )
    parser.add_argument(
        "--baseline_file",
        type=str,
        default="",
        help="无记忆 baseline 文件路径，用于计算 Personalization Gain",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=3,
        help="per-user 相关系数计算的最小样本数阈值（默认 3）",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="将评估结果保存为 JSON 文件路径（留空则不保存）",
    )
    args = parser.parse_args()

    all_results: dict[str, dict] = {}

    # ── 单文件评估 ────────────────────────────────────────────────────────────
    if args.result_file:
        name = os.path.splitext(os.path.basename(args.result_file))[0]
        res = evaluate_single(args.result_file, name, min_samples=args.min_samples)
        all_results[name] = res

        # 计算个性化增益（若提供 baseline）
        if args.baseline_file:
            mem_records = load_records(args.result_file)
            base_records = load_records(args.baseline_file)
            pg = compute_personalization_gain(mem_records, base_records)
            pu_pg = per_user_personalization_gain(mem_records, base_records)
            print_personalization_gain(pg, pu_pg)
            res["personalization_gain"] = {**pg, **pu_pg}

    # ── 多文件对比 ────────────────────────────────────────────────────────────
    if args.result_files:
        named_files: list[tuple[str, str]] = []
        for item in args.result_files:
            if "=" in item:
                name, path = item.split("=", 1)
            else:
                name = os.path.splitext(os.path.basename(item))[0]
                path = item
            named_files.append((name, path))

        for name, path in named_files:
            res = evaluate_single(path, name, min_samples=args.min_samples)
            all_results[name] = res

        if len(all_results) > 1:
            print_comparison_table(all_results)
            print_boundary_comparison_table(all_results)

        # 自动检测 baseline（with_memory=False 的记录）
        baseline_name = next(
            (n for n, p in named_files if "no_memory" in n or "baseline" in n),
            None,
        )
        mem_name = next(
            (n for n, p in named_files if "with_memory" in n or "memory" in n and n != baseline_name),
            None,
        )
        if baseline_name and mem_name:
            mem_records = load_records(dict(named_files)[mem_name])
            base_records = load_records(dict(named_files)[baseline_name])
            pg = compute_personalization_gain(mem_records, base_records)
            pu_pg = per_user_personalization_gain(mem_records, base_records)
            print_personalization_gain(pg, pu_pg)

    # ── 保存 JSON ─────────────────────────────────────────────────────────────
    if args.output_json and all_results:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as fp:
            json.dump(all_results, fp, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存至: {args.output_json}")


if __name__ == "__main__":
    main()
