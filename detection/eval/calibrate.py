"""
Post-hoc 校准：用每个 (user, target_task, model) 块的【用户历史分数分布】
（来自 memory_cache 中的 `score_distribution` / `avg_satisfaction_score`）
对该块内的模型预测做 scale 对齐。

动机
----
诊断（reports/diagnose_confusion.md、reports/anchor_and_diagnostics.md）显示：
  - GPT-4o-mini 预测分布塌缩到 4（59% 预测为 4 vs 真实 37.8%）
  - Qwen3-8B 系统性低估（平均 bias −0.20）
两者**相对排序**尚可（Pearson > 0），但**绝对刻度**偏离用户真实评分风格。
该模块按用户的历史分数分布做"rank → score"重映射，零额外 API 开销。

方法
----
三种策略（--method）：

1. `identity`  不变（sanity check）
2. `mean_shift`  每块预测整体平移，使 pred_mean ≈ hist_mean。
                 pred_new = pred - pred_mean + hist_mean（再 clip 到 [1,5]、round）。
3. `cdf`        每块内对预测排名（小→大），映射到历史 CDF 的分位点。
                若块内有 n 条预测，给第 i 条分位点 q=(i+0.5)/n；
                然后在历史累计分布上找最小 s* 使 P(Score<=s*) ≥ q。
                同分的预测按 turn_idx 做稳定排序。

注意：
  - 历史分布来自训练集（source_tasks），测试集是 target_task——跨任务迁移
    的前提是【用户评分风格】在任务间一致。
  - 若某块 n_history_turns < min_history_turns 或块内预测数 < 2，降级为 identity。

运行方式（从 detection/ 目录）：
  python eval/calibrate.py \\
      --input_jsonl outputs/personalized/gpt-4o-mini_test_none_v2.jsonl \\
      --memory_cache_dir outputs/personalized/memory_cache \\
      --method cdf \\
      --output_jsonl outputs/personalized/gpt-4o-mini_test_none_v2_calCDF.jsonl
"""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from collections import defaultdict

from loguru import logger


# ──────────────────────────────────────────────────────────────────────────────
# IO
# ──────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def save_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_memory_cache(cache_dir: str, user: str, task: str, model: str) -> dict | None:
    fn = f"{user}__{task}__{model.replace('/', '_')}.json"
    path = os.path.join(cache_dir, fn)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# 校准策略
# ──────────────────────────────────────────────────────────────────────────────

def _clip_round(x: float) -> int:
    return max(1, min(5, int(round(x))))


def calibrate_mean_shift(preds: list[float], hist_mean: float) -> list[int]:
    if not preds:
        return []
    pred_mean = sum(preds) / len(preds)
    delta = hist_mean - pred_mean
    return [_clip_round(p + delta) for p in preds]


def _hist_cdf(score_dist: dict[str, int]) -> list[tuple[int, float]]:
    """返回 [(score, cdf_at_score)]，按 score 升序。"""
    total = sum(score_dist.get(f"score_{s}", 0) for s in range(1, 6))
    if total == 0:
        return []
    cum = 0
    out: list[tuple[int, float]] = []
    for s in range(1, 6):
        cum += score_dist.get(f"score_{s}", 0)
        out.append((s, cum / total))
    return out


def _inv_cdf(cdf: list[tuple[int, float]], q: float) -> int:
    """找最小 s* 使 P(Score<=s*) >= q。"""
    for s, p in cdf:
        if p >= q:
            return s
    return cdf[-1][0]


def calibrate_cdf(
    preds_with_order: list[tuple[float, int]],
    score_dist: dict[str, int],
) -> list[int]:
    """
    输入: [(pred, turn_idx), ...] 顺序等于 records 顺序
    输出: 与输入同长的校准后整数分
    """
    if not preds_with_order:
        return []
    cdf = _hist_cdf(score_dist)
    if not cdf:
        return [_clip_round(p) for p, _ in preds_with_order]

    n = len(preds_with_order)
    # 附带原始位置，按 (pred, turn_idx) 升序稳定排序
    indexed = [(p, ti, i) for i, (p, ti) in enumerate(preds_with_order)]
    indexed.sort(key=lambda x: (x[0], x[1]))

    new_scores: list[int] = [0] * n
    for rank, (_, _, orig_i) in enumerate(indexed):
        q = (rank + 0.5) / n
        new_scores[orig_i] = _inv_cdf(cdf, q)
    return new_scores


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def calibrate_records(
    records: list[dict],
    memory_cache_dir: str,
    method: str,
    min_history_turns: int = 5,
    min_block_size: int = 2,
) -> tuple[list[dict], dict]:
    """
    按 (user, target_task, model) 分块校准。
    返回 (校准后记录, 统计信息)
    """
    # 分块
    blocks: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        blocks[(r["user"], r["target_task"], r["model"])].append(i)

    out = [dict(r) for r in records]
    stats = {
        "n_blocks": len(blocks),
        "n_calibrated": 0,
        "n_fallback_identity": 0,
        "fallback_reasons": defaultdict(int),
        "method": method,
    }

    for (user, task, model), idxs in blocks.items():
        mem = load_memory_cache(memory_cache_dir, user, task, model)
        if mem is None:
            stats["fallback_reasons"]["no_memory_cache"] += 1
            stats["n_fallback_identity"] += 1
            continue

        n_hist = int(mem.get("n_history_turns", 0))
        score_dist = mem.get("score_distribution") or {}
        hist_mean = float(mem.get("avg_satisfaction_score", 0.0))

        if n_hist < min_history_turns:
            stats["fallback_reasons"]["thin_history"] += 1
            stats["n_fallback_identity"] += 1
            continue
        if len(idxs) < min_block_size:
            stats["fallback_reasons"]["small_block"] += 1
            stats["n_fallback_identity"] += 1
            continue
        if method == "identity":
            stats["n_fallback_identity"] += 1
            continue

        if method == "mean_shift":
            preds = [float(records[i]["pred_score"]) for i in idxs]
            new_preds = calibrate_mean_shift(preds, hist_mean)
        elif method == "cdf":
            preds_with_order = [
                (float(records[i]["pred_score"]), int(records[i].get("turn_idx", 0)))
                for i in idxs
            ]
            new_preds = calibrate_cdf(preds_with_order, score_dist)
        else:
            raise ValueError(f"未知 method: {method}")

        for local_i, global_i in enumerate(idxs):
            out[global_i]["pred_score_raw"] = records[global_i]["pred_score"]
            out[global_i]["pred_score"] = new_preds[local_i]
            out[global_i]["calibration_method"] = method
        stats["n_calibrated"] += 1

    stats["fallback_reasons"] = dict(stats["fallback_reasons"])
    return out, stats


def main() -> None:
    parser = ArgumentParser(description="个性化满意度预测后处理校准")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, default="")
    parser.add_argument(
        "--memory_cache_dir",
        type=str,
        default="outputs/personalized/memory_cache",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["identity", "mean_shift", "cdf"],
        default="cdf",
    )
    parser.add_argument("--min_history_turns", type=int, default=5)
    parser.add_argument("--min_block_size", type=int, default=2)
    args = parser.parse_args()

    records = load_jsonl(args.input_jsonl)
    logger.info(f"加载记录: {len(records)} from {args.input_jsonl}")

    out, stats = calibrate_records(
        records,
        memory_cache_dir=args.memory_cache_dir,
        method=args.method,
        min_history_turns=args.min_history_turns,
        min_block_size=args.min_block_size,
    )
    logger.info(f"校准统计: {json.dumps(stats, ensure_ascii=False)}")

    output_path = args.output_jsonl
    if not output_path:
        base, ext = os.path.splitext(args.input_jsonl)
        tag = {"identity": "calID", "mean_shift": "calMS", "cdf": "calCDF"}[args.method]
        output_path = f"{base}_{tag}{ext}"

    save_jsonl(out, output_path)
    logger.info(f"已保存: {output_path}")


if __name__ == "__main__":
    main()
