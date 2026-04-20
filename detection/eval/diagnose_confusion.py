"""
满意度预测诊断脚本：5×5 混淆矩阵 + 分数分布 + 用户级均值偏差

用途
----
在不跑全量推理的前提下，从已有 JSONL 结果文件快速回答：
  1. 模型把哪类 gold 分数系统性误判成哪一档？
     → 5×5 混淆矩阵（行 = gold，列 = pred），行归一化百分比
  2. 模型整体分数分布是否与 gold 匹配？
     → 边缘分布对比（gold vs pred）
  3. 不同用户间的校准偏差：模型在该用户上的 mean_pred vs mean_gold
     → 用户级偏差直方图（over-predict / under-predict 分布）

可输入一个或多个结果文件，多文件时并列输出对比。

示例
----
    python eval/diagnose_confusion.py \\
        --result_files \\
            gpt4o_none=outputs/personalized/gpt-4o-mini_test_none_v2.jsonl \\
            qwen3_none=outputs/personalized/Qwen_Qwen3-8B_test_none.jsonl \\
        --output_md reports/diagnose_confusion.md
"""

from __future__ import annotations

import json
import math
import os
from argparse import ArgumentParser
from collections import defaultdict

from loguru import logger


# ──────────────────────────────────────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────────────────────────────────────

def load_records(path: str) -> list[dict]:
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
            except json.JSONDecodeError:
                continue
            if r.get("gold_score") is None or r.get("pred_score") is None:
                continue
            records.append(r)
    return records


# ──────────────────────────────────────────────────────────────────────────────
# 指标计算
# ──────────────────────────────────────────────────────────────────────────────

def _clamp_int(v, lo: int = 1, hi: int = 5) -> int:
    iv = int(round(float(v)))
    return max(lo, min(hi, iv))


def confusion_matrix(records: list[dict]) -> tuple[list[list[int]], list[int], list[int]]:
    """返回 (5x5 matrix, gold_marginal, pred_marginal), 索引 1..5 放在 0..4。"""
    m = [[0] * 5 for _ in range(5)]
    gm = [0] * 5
    pm = [0] * 5
    for r in records:
        g = _clamp_int(r["gold_score"]) - 1
        p = _clamp_int(r["pred_score"]) - 1
        m[g][p] += 1
        gm[g] += 1
        pm[p] += 1
    return m, gm, pm


def per_user_bias(records: list[dict]) -> list[dict]:
    """每个用户的 mean_pred - mean_gold 偏差列表。"""
    by_user: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for r in records:
        u = str(r.get("user", "unknown"))
        by_user[u][0].append(float(r["gold_score"]))
        by_user[u][1].append(float(r["pred_score"]))
    out = []
    for u, (g, p) in by_user.items():
        if not g:
            continue
        mg = sum(g) / len(g)
        mp = sum(p) / len(p)
        out.append({
            "user": u,
            "n": len(g),
            "mean_gold": mg,
            "mean_pred": mp,
            "bias": mp - mg,   # + 代表整体高估
        })
    out.sort(key=lambda x: x["bias"])
    return out


def summarize_bias(biases: list[dict]) -> dict:
    if not biases:
        return {}
    vals = [b["bias"] for b in biases]
    n = len(vals)
    mean_bias = sum(vals) / n
    var = sum((v - mean_bias) ** 2 for v in vals) / n
    std = math.sqrt(var)
    n_over = sum(1 for v in vals if v > 0.2)
    n_under = sum(1 for v in vals if v < -0.2)
    n_neutral = n - n_over - n_under
    abs_bias = sum(abs(v) for v in vals) / n
    return {
        "n_users": n,
        "mean_bias": mean_bias,
        "std_bias": std,
        "mean_abs_bias": abs_bias,
        "n_over_predict": n_over,
        "n_under_predict": n_under,
        "n_neutral": n_neutral,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 格式化输出
# ──────────────────────────────────────────────────────────────────────────────

def format_confusion_block(name: str, records: list[dict]) -> str:
    """用 markdown 格式返回单文件的诊断块（供日志打印 + 报告写入）。"""
    m, gm, pm = confusion_matrix(records)
    total = sum(gm)
    lines: list[str] = []
    lines.append(f"### {name}")
    lines.append("")
    lines.append(f"- 总样本数：{total}")
    lines.append(f"- 用户数：{len({r.get('user') for r in records})}")
    lines.append("")

    # ── 5x5 混淆矩阵（行归一化 %）──────────────────────────────────────────────
    lines.append("**混淆矩阵（行=gold，列=pred；每格：count / 行百分比）：**")
    lines.append("")
    lines.append("| gold \\ pred | 1 | 2 | 3 | 4 | 5 | 行计 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for gi in range(5):
        row_total = gm[gi]
        cells = []
        for pj in range(5):
            c = m[gi][pj]
            pct = (c / row_total * 100) if row_total else 0.0
            cells.append(f"{c} / {pct:.1f}%" if row_total else "0 / -")
        lines.append(f"| **{gi + 1}** | " + " | ".join(cells) + f" | {row_total} |")
    lines.append(f"| 列计 | {pm[0]} | {pm[1]} | {pm[2]} | {pm[3]} | {pm[4]} | {total} |")
    lines.append("")

    # ── 边缘分布对比 ─────────────────────────────────────────────────────────
    lines.append("**边缘分布对比（百分比）：**")
    lines.append("")
    lines.append("| 分数 | gold% | pred% | Δ (pred-gold) |")
    lines.append("|---|---:|---:|---:|")
    for i in range(5):
        gp = gm[i] / total * 100 if total else 0.0
        pp = pm[i] / total * 100 if total else 0.0
        lines.append(f"| {i + 1} | {gp:.1f}% | {pp:.1f}% | {pp - gp:+.1f}% |")
    lines.append("")

    # ── 全局偏差和 MAE 分解 ──────────────────────────────────────────────────
    if total:
        avg_gold = sum((i + 1) * gm[i] for i in range(5)) / total
        avg_pred = sum((i + 1) * pm[i] for i in range(5)) / total
        lines.append(
            f"**整体均值：** gold={avg_gold:.3f}，pred={avg_pred:.3f}，"
            f"系统性偏差={avg_pred - avg_gold:+.3f}"
        )
        lines.append("")

    # ── 每 gold 分数上的 MAE ──────────────────────────────────────────────────
    lines.append("**各 gold 分数上的 MAE（模型在哪一档错最多）：**")
    lines.append("")
    lines.append("| gold | n | MAE | 主要误判去向 |")
    lines.append("|---|---:|---:|---|")
    for gi in range(5):
        row_total = gm[gi]
        if not row_total:
            lines.append(f"| {gi + 1} | 0 | - | - |")
            continue
        mae = sum(m[gi][pj] * abs((pj + 1) - (gi + 1)) for pj in range(5)) / row_total
        # 找最大误判（非对角）
        off_diag = [(pj, m[gi][pj]) for pj in range(5) if pj != gi]
        off_diag.sort(key=lambda x: x[1], reverse=True)
        top_misclass = off_diag[0] if off_diag else (None, 0)
        if top_misclass[1] > 0:
            tp_pct = top_misclass[1] / row_total * 100
            top_desc = f"→ {top_misclass[0] + 1} ({tp_pct:.1f}%)"
        else:
            top_desc = "-"
        lines.append(f"| {gi + 1} | {row_total} | {mae:.3f} | {top_desc} |")
    lines.append("")

    # ── 用户级偏差 ───────────────────────────────────────────────────────────
    biases = per_user_bias(records)
    bsum = summarize_bias(biases)
    if bsum:
        lines.append(
            "**用户级均值偏差（bias = mean_pred - mean_gold，阈值 ±0.2）：**"
        )
        lines.append("")
        lines.append(
            f"- 用户数：{bsum['n_users']}，"
            f"平均偏差：{bsum['mean_bias']:+.3f}，"
            f"偏差标准差：{bsum['std_bias']:.3f}，"
            f"平均绝对偏差：{bsum['mean_abs_bias']:.3f}"
        )
        lines.append(
            f"- 高估用户（bias > +0.2）：{bsum['n_over_predict']}；"
            f"低估用户（bias < −0.2）：{bsum['n_under_predict']}；"
            f"基本对齐：{bsum['n_neutral']}"
        )
        # 列出偏差最极端的 5 个用户（每方向各最多 5）
        worst_over = [b for b in biases if b["bias"] > 0.2][-5:]
        worst_under = [b for b in biases if b["bias"] < -0.2][:5]
        if worst_over or worst_under:
            lines.append("")
            lines.append("  偏差最极端的用户（展示 bias 最大/最小各 5）：")
            for b in worst_under:
                lines.append(
                    f"  - {b['user']}  n={b['n']}  "
                    f"gold={b['mean_gold']:.2f}  pred={b['mean_pred']:.2f}  "
                    f"bias={b['bias']:+.2f}"
                )
            for b in reversed(worst_over):
                lines.append(
                    f"  - {b['user']}  n={b['n']}  "
                    f"gold={b['mean_gold']:.2f}  pred={b['mean_pred']:.2f}  "
                    f"bias={b['bias']:+.2f}"
                )
        lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = ArgumentParser(description="满意度预测诊断：混淆矩阵 + 分数分布 + 用户级偏差")
    parser.add_argument(
        "--result_files",
        type=str,
        nargs="+",
        required=True,
        metavar="NAME=PATH",
        help="结果文件列表，格式 name=path",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default="",
        help="将完整报告保存为 Markdown 文件",
    )
    args = parser.parse_args()

    all_blocks: list[str] = [
        "# 满意度预测诊断报告",
        "",
        "- 内容：5×5 混淆矩阵 + 分数边缘分布 + 每 gold 分数的 MAE 分解 + 用户级均值偏差",
        "- 数据：从已有 JSONL 结果文件直接计算，不重新推理",
        "",
    ]

    for item in args.result_files:
        if "=" not in item:
            logger.error(f"格式错误，需要 name=path，实际：{item}")
            continue
        name, path = item.split("=", 1)
        records = load_records(path)
        if not records:
            logger.warning(f"跳过（无有效记录）：{path}")
            continue
        logger.info(f"  {name:<30} n_records={len(records)}  file={path}")
        block = format_confusion_block(name, records)
        # 打印到终端
        logger.info("\n" + block)
        all_blocks.append(f"## 来源：`{path}`")
        all_blocks.append("")
        all_blocks.append(block)
        all_blocks.append("---")
        all_blocks.append("")

    # ── 保存 md 报告 ──────────────────────────────────────────────────────────
    if args.output_md:
        os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
        with open(args.output_md, "w", encoding="utf-8") as fp:
            fp.write("\n".join(all_blocks))
        logger.info(f"报告已保存：{args.output_md}")


if __name__ == "__main__":
    main()
