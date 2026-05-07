from __future__ import annotations

import json
import os
import random
from argparse import ArgumentParser

from loguru import logger

from lib.data_split import split_by_user_group_shuffle_split
from lib.metric_statistics import get_satisfaction_data

from .constants import DSAT_LABEL, SAT_LABEL
from .data import format_conversation, preprocess_to_rows
from .embeddings import build_rubric_feature_vec, get_embeddings, train_and_eval_classifier
from .metrics import compute_metrics, print_metrics
from .rubrics import extract_rubric_candidates, summarize_rubrics
from .scoring import score_rows, score_test_set


def main(
    model: str = "gpt-4o",
    k_rubrics: int = 10,
    max_extract_per_label: int = 150,
    max_workers: int = 8,
    output_dir: str = "./outputs/spur",
    limit_test: int = 0,
    seed: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)

    # ── 数据加载与切分 ──────────────────────────────────────────────────────
    logger.info("加载数据...")
    data_list = get_satisfaction_data()
    rows = preprocess_to_rows(data_list)
    logger.info(
        f"总样本: {len(rows)}  SAT={sum(1 for r in rows if r['binary_label']==SAT_LABEL)}"
        f"  DSAT={sum(1 for r in rows if r['binary_label']==DSAT_LABEL)}"
    )

    users = [r["user"] for r in rows]
    train_idx, valid_idx, test_idx = split_by_user_group_shuffle_split(
        users, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=seed
    )
    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]
    logger.info(f"Train={len(train_rows)}, Test={len(test_rows)}")

    if limit_test > 0:
        rng = random.Random(seed)
        test_rows = rng.sample(test_rows, min(limit_test, len(test_rows)))
        logger.info(f"--limit_test 限制测试集为 {len(test_rows)} 条")

    # ── Phase 1: Supervised Extraction ─────────────────────────────────────
    p1_cache = os.path.join(output_dir, "phase1_candidates.json")
    candidates = extract_rubric_candidates(
        train_rows,
        model=model,
        cache_file=p1_cache,
        max_workers=max_workers,
        max_per_label=max_extract_per_label,
    )

    # ── Phase 2: Rubric Summarization ──────────────────────────────────────
    p2_cache = os.path.join(output_dir, f"phase2_rubrics_k{k_rubrics}.json")
    rubrics = summarize_rubrics(
        candidates,
        model=model,
        k=k_rubrics,
        cache_file=p2_cache,
    )

    # ── Phase 3: Scoring ───────────────────────────────────────────────────
    p3_cache = os.path.join(output_dir, f"phase3_results_k{k_rubrics}.jsonl")
    results = score_test_set(
        test_rows,
        rubrics=rubrics,
        model=model,
        cache_file=p3_cache,
        max_workers=max_workers,
    )

    # ── 评估 ────────────────────────────────────────────────────────────────
    metrics = compute_metrics(results)
    print_metrics(metrics, header="SPUR 满意度二分类评估结果")

    metrics_path = os.path.join(output_dir, f"metrics_k{k_rubrics}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"指标已保存至: {metrics_path}")


def parse_args():
    p = ArgumentParser(description="SPUR 用户满意度二分类估计 (Lin et al., ACL 2024)")
    p.add_argument("--model", type=str, default="gpt-4o",
                   help="LLM 模型名称（需与 api_config.json 中的 base_url 兼容）")
    p.add_argument("--k_rubrics", type=int, default=10,
                   help="每个标签归纳的 rubric 数量（论文默认 10）")
    p.add_argument("--max_extract_per_label", type=int, default=150,
                   help="Phase 1 每个标签最多使用的训练样本数（避免 API 调用过多）")
    p.add_argument("--max_workers", type=int, default=8,
                   help="并行 API 调用线程数")
    p.add_argument("--output_dir", type=str, default="./outputs/spur",
                   help="缓存和结果输出目录")
    p.add_argument("--limit_test", type=int, default=0,
                   help="调试用：限制测试集样本数（0=全量）")
    p.add_argument("--seed", type=int, default=42)

    # 跳过某些阶段（用于断点续跑）
    p.add_argument("--skip_phase1", action="store_true",
                   help="跳过 Phase 1，直接从缓存加载候选 rubric（需缓存文件存在）")
    p.add_argument("--skip_phase2", action="store_true",
                   help="跳过 Phase 2，直接从缓存加载汇总 rubric（需缓存文件存在）")
    p.add_argument("--only_eval", action="store_true",
                   help="仅对已有 Phase 3 结果重新计算指标，不调用 LLM")

    # Phase 4：Text embedding 辅助分类器
    p.add_argument("--use_embeddings", action="store_true",
                   help="启用 Phase 4：用 text embedding + rubric 特征训练 LogisticRegression 分类器")
    p.add_argument("--embedding_model", type=str, default="text-embedding-ada-002",
                   help="OpenAI embedding 模型名（默认 text-embedding-ada-002）")
    p.add_argument("--embedding_batch_size", type=int, default=64,
                   help="每次调用 embedding API 的批量大小")
    return p.parse_args()


def run_cli() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据（所有模式都需要）
    logger.info("加载数据...")
    data_list = get_satisfaction_data()
    rows = preprocess_to_rows(data_list)
    users = [r["user"] for r in rows]
    train_idx, valid_idx, test_idx = split_by_user_group_shuffle_split(
        users, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=args.seed
    )
    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]
    logger.info(f"Train={len(train_rows)}, Test={len(test_rows)}")

    if args.limit_test > 0:
        rng = random.Random(args.seed)
        test_rows = rng.sample(test_rows, min(args.limit_test, len(test_rows)))
        logger.info(f"测试集限制为 {len(test_rows)} 条")

    p1_cache = os.path.join(args.output_dir, "phase1_candidates.json")
    p2_cache = os.path.join(args.output_dir, f"phase2_rubrics_k{args.k_rubrics}.json")
    p3_cache = os.path.join(args.output_dir, f"phase3_results_k{args.k_rubrics}.jsonl")

    # ── only_eval：直接从 Phase 3 结果计算指标 ──────────────────────────────
    if args.only_eval:
        if not os.path.exists(p3_cache):
            logger.error(f"Phase 3 缓存不存在: {p3_cache}")
            raise SystemExit(1)
        results = []
        with open(p3_cache) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        logger.info(f"从缓存加载 {len(results)} 条结果")
        metrics = compute_metrics(results)
        print_metrics(metrics, header="SPUR 评估结果（仅重新计算指标）")
        raise SystemExit(0)

    # ── Phase 1 ─────────────────────────────────────────────────────────────
    if args.skip_phase1:
        if not os.path.exists(p1_cache):
            logger.error(f"--skip_phase1 指定但缓存不存在: {p1_cache}")
            raise SystemExit(1)
        logger.info(f"[Phase 1] 跳过，从缓存加载: {p1_cache}")
        with open(p1_cache) as f:
            candidates = json.load(f)
    else:
        candidates = extract_rubric_candidates(
            train_rows,
            model=args.model,
            cache_file=p1_cache,
            max_workers=args.max_workers,
            max_per_label=args.max_extract_per_label,
        )

    # ── Phase 2 ─────────────────────────────────────────────────────────────
    if args.skip_phase2:
        if not os.path.exists(p2_cache):
            logger.error(f"--skip_phase2 指定但缓存不存在: {p2_cache}")
            raise SystemExit(1)
        logger.info(f"[Phase 2] 跳过，从缓存加载: {p2_cache}")
        with open(p2_cache) as f:
            rubrics = json.load(f)
    else:
        rubrics = summarize_rubrics(
            candidates,
            model=args.model,
            k=args.k_rubrics,
            cache_file=p2_cache,
        )

    # ── Phase 3：对测试集评分 ──────────────────────────────────────────────
    test_scored = score_rows(
        test_rows,
        rubrics=rubrics,
        model=args.model,
        cache_file=p3_cache,
        max_workers=args.max_workers,
        desc="Phase 3 (test)",
    )

    # ── 评估（直接 LLM 判断）────────────────────────────────────────────────
    metrics = compute_metrics(test_scored)
    print_metrics(metrics, header="SPUR (直接 LLM 判断)")

    all_metrics: dict[str, dict] = {"spur_direct": metrics}

    # ── Phase 4（可选）：embedding + 分类器 ──────────────────────────────────
    if args.use_embeddings:
        # 4a：对训练集也做 rubric 评分（分类器需要训练标签+特征）
        p3_train_cache = os.path.join(
            args.output_dir, f"phase3_train_results_k{args.k_rubrics}.jsonl"
        )
        train_scored = score_rows(
            train_rows,
            rubrics=rubrics,
            model=args.model,
            cache_file=p3_train_cache,
            max_workers=args.max_workers,
            desc="Phase 3 (train)",
        )

        # 4b：获取 embedding
        train_texts = [format_conversation(r) for r in train_rows]
        test_texts = [format_conversation(r) for r in test_rows]

        train_emb_cache = os.path.join(args.output_dir, "embeddings_train.npz")
        test_emb_cache = os.path.join(args.output_dir, "embeddings_test.npz")

        train_emb = get_embeddings(
            train_texts,
            model=args.embedding_model,
            cache_file=train_emb_cache,
            batch_size=args.embedding_batch_size,
        )
        test_emb = get_embeddings(
            test_texts,
            model=args.embedding_model,
            cache_file=test_emb_cache,
            batch_size=args.embedding_batch_size,
        )

        # 4c：构建 rubric 特征向量
        k = args.k_rubrics
        train_rubric_feats = build_rubric_feature_vec(train_scored, k)
        test_rubric_feats = build_rubric_feature_vec(test_scored, k)

        train_labels = [1 if r["gold_label"] == SAT_LABEL else 0 for r in train_scored]
        test_labels = [1 if r["gold_label"] == SAT_LABEL else 0 for r in test_scored]

        # 4d：训练并评估分类器（rubric_only / embedding_only / combined）
        clf_metrics = train_and_eval_classifier(
            train_rubric_feats, train_emb, train_labels,
            test_rubric_feats, test_emb, test_labels,
            test_scored,
        )
        for variant, m in clf_metrics.items():
            print_metrics(m, header=f"SPUR + Classifier ({variant})")
            all_metrics[f"clf_{variant}"] = m

    # ── 保存全部指标 ─────────────────────────────────────────────────────────
    metrics_path = os.path.join(args.output_dir, f"metrics_k{args.k_rubrics}.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"指标已保存至: {metrics_path}")
