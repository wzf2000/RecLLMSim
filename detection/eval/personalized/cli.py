from __future__ import annotations

import json
import os
from argparse import ArgumentParser

from loguru import logger

from .gain import compute_personalization_gain, per_user_personalization_gain, print_personalization_gain
from .io import load_records
from .runner import evaluate_single
from .tables import print_boundary_comparison_table, print_comparison_table


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


