"""SPUR baseline adapted to the personalized cross-task split.

Default mode is SPUR-direct:
1. Extract SAT/DSAT rubric candidates from train-split users.
2. Summarize global SAT/DSAT rubrics.
3. Apply rubrics to personalized test turns.
4. Export eval/personalized.py-compatible JSONL with pred_score in {3, 4}.

Optional classifier variants are implemented but not used by default:
rubric_lr, embedding_lr, combined.
"""

from __future__ import annotations

import json
import os
import random
import sys
from argparse import ArgumentParser

from loguru import logger

_DETECTION_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _DETECTION_DIR not in sys.path:
    sys.path.insert(0, _DETECTION_DIR)

from eval.spur.constants import DSAT_LABEL, SAT_LABEL
from eval.spur.data import format_conversation, format_profile
from eval.spur.embeddings import build_rubric_feature_vec, get_embeddings
from eval.spur.llm import configure_client
from eval.spur.metrics import compute_metrics, print_metrics
from eval.spur.rubrics import extract_rubric_candidates, summarize_rubrics
from eval.spur.scoring import score_rows
from lib.personalized_data import PersonalizedSample, build_personalized_samples, dataset_stats
from lib.satisfaction_constants import normalize_reason_for_score


def _row_from_turn(
    sample: PersonalizedSample,
    session,
    session_file: str,
    turn_idx: int,
    history_window: list[str],
    assistant_reply: str,
) -> dict:
    score = int(session.satisfaction_scores[turn_idx])
    reason = (
        session.dissatisfaction_reasons[turn_idx]
        if turn_idx < len(session.dissatisfaction_reasons)
        else ""
    )
    return {
        "sample_id": f"{sample.user}__{sample.target_task}__{session_file}__turn_{turn_idx}",
        "persona": format_profile(sample.profile),
        "task_context": session.task_context,
        "history": "\n".join(history_window),
        "assistant_reply": assistant_reply,
        "gold_score": score,
        "gold_reason": normalize_reason_for_score(score, reason),
        "binary_label": SAT_LABEL if score >= 4 else DSAT_LABEL,
        "user": sample.user,
        "target_task": sample.target_task,
        "target_file": session_file,
        "turn_idx": turn_idx,
        "source_chat_model": session.chat_model,
    }


def personalized_samples_to_spur_rows(
    samples: list[PersonalizedSample],
    history_window_size: int,
) -> list[dict]:
    rows: list[dict] = []
    for sample in samples:
        for session in sample.target_sessions:
            session_file = os.path.basename(session.file_path)
            history_window: list[str] = []
            assistant_idx = 0
            for utt in session.history:
                if utt["role"] == "assistant":
                    rows.append(_row_from_turn(
                        sample=sample,
                        session=session,
                        session_file=session_file,
                        turn_idx=assistant_idx,
                        history_window=history_window,
                        assistant_reply=utt["content"],
                    ))
                    assistant_idx += 1
                history_window.append(f'{utt["role"]}：{utt["content"]}')
                while len(history_window) > history_window_size:
                    history_window.pop(0)
    return rows


def _load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def _save_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def spur_results_to_personalized_records(
    rows: list[dict],
    scored: list[dict],
    model: str,
    variant: str,
) -> list[dict]:
    records: list[dict] = []
    for row, result in zip(rows, scored):
        pred_label = result.get("pred_label", DSAT_LABEL)
        pred_score = 4 if pred_label == SAT_LABEL else 3
        records.append({
            "sample_id": row["sample_id"],
            "user": row["user"],
            "target_task": row["target_task"],
            "target_file": row["target_file"],
            "turn_idx": row["turn_idx"],
            "model": f"spur_{variant}:{model}",
            "with_memory": False,
            "memory_version": "none",
            "memory_update_mode": "none",
            "turn_eval_prompt_version": f"spur_{variant}",
            "source_chat_model": row.get("source_chat_model", "unknown"),
            "gold_score": row["gold_score"],
            "gold_reason": row["gold_reason"],
            "pred_score": pred_score,
            "pred_reason": "满意" if pred_score >= 4 else "其它",
            "reason_prediction": "满意" if pred_score >= 4 else "其它",
            "spur_pred_label": pred_label,
            "spur_confidence": float(result.get("confidence", 0.5)),
            "spur_sat_matches": result.get("sat_matches", []),
            "spur_dsat_matches": result.get("dsat_matches", []),
            "spur_reason": result.get("reason", ""),
            "parse_ok": bool(result.get("parse_ok", False)),
        })
    return records


def _train_lr_variant(
    variant: str,
    train_rows: list[dict],
    test_rows: list[dict],
    train_scored: list[dict],
    test_scored: list[dict],
    k_rubrics: int,
    embedding_model: str,
    embedding_batch_size: int,
    output_dir: str,
    seed: int,
) -> list[dict]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    train_rubric = build_rubric_feature_vec(train_scored, k_rubrics)
    test_rubric = build_rubric_feature_vec(test_scored, k_rubrics)

    train_emb = None
    test_emb = None
    if variant in {"embedding_lr", "combined"}:
        train_texts = [format_conversation(r) for r in train_rows]
        test_texts = [format_conversation(r) for r in test_rows]
        train_emb = get_embeddings(
            train_texts,
            model=embedding_model,
            cache_file=os.path.join(output_dir, f"embeddings_train_{embedding_model}.npz"),
            batch_size=embedding_batch_size,
        )
        test_emb = get_embeddings(
            test_texts,
            model=embedding_model,
            cache_file=os.path.join(output_dir, f"embeddings_test_{embedding_model}.npz"),
            batch_size=embedding_batch_size,
        )

    if variant == "rubric_lr":
        x_train = train_rubric
        x_test = test_rubric
    elif variant == "embedding_lr":
        if train_emb is None or test_emb is None:
            raise RuntimeError("embedding_lr requires embeddings")
        x_train = train_emb
        x_test = test_emb
    elif variant == "combined":
        if train_emb is None or test_emb is None:
            raise RuntimeError("combined requires embeddings")
        x_train = np.concatenate([train_rubric, train_emb], axis=1)
        x_test = np.concatenate([test_rubric, test_emb], axis=1)
    else:
        raise ValueError(f"Unsupported LR variant: {variant}")

    y_train = [1 if r["binary_label"] == SAT_LABEL else 0 for r in train_rows]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test).tolist()
    prob = clf.predict_proba(x_test)[:, 1].tolist()
    out: list[dict] = []
    for row, label, p_sat in zip(test_rows, pred, prob):
        pred_label = SAT_LABEL if int(label) == 1 else DSAT_LABEL
        out.append({
            "gold_score": row["gold_score"],
            "gold_label": row["binary_label"],
            "pred_label": pred_label,
            "confidence": float(p_sat if pred_label == SAT_LABEL else 1.0 - p_sat),
            "sat_matches": [],
            "dsat_matches": [],
            "reason": f"{variant} logistic regression",
            "parse_ok": True,
        })
    return out


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Run SPUR on the personalized split")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--variant", type=str, default="direct",
                        choices=["direct", "rubric_lr", "embedding_lr", "combined"])
    parser.add_argument("--k_rubrics", type=int, default=10)
    parser.add_argument("--max_extract_per_label", type=int, default=150)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="outputs/spur_personalized")
    parser.add_argument("--output_jsonl", type=str, default="")
    parser.add_argument("--metrics_json", type=str, default="")
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.2)
    parser.add_argument("--min_history_sessions", type=int, default=1)
    parser.add_argument("--target_tasks", type=str, nargs="+", default=None)
    parser.add_argument("--history_window_size", type=int, default=5)
    parser.add_argument("--limit_test", type=int, default=0)
    parser.add_argument("--limit_train", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_phase1", action="store_true")
    parser.add_argument("--skip_phase2", action="store_true")
    parser.add_argument("--only_eval", action="store_true")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-ada-002")
    parser.add_argument("--embedding_batch_size", type=int, default=64)
    return parser


def main() -> None:
    args = parse_args().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    configure_client(base_url=args.base_url, api_key=args.api_key)

    if not args.output_jsonl:
        args.output_jsonl = os.path.join(
            args.output_dir,
            f"spur_{args.variant}_{args.model}_personalized_test.jsonl".replace("/", "_"),
        )
    if not args.metrics_json:
        base, _ = os.path.splitext(args.output_jsonl)
        args.metrics_json = base + "_metrics.json"

    if args.only_eval:
        records = _load_jsonl(args.output_jsonl)
        direct_like = [
            {
                "gold_score": r["gold_score"],
                "gold_label": SAT_LABEL if int(r["gold_score"]) >= 4 else DSAT_LABEL,
                "pred_label": r.get("spur_pred_label", SAT_LABEL if int(r["pred_score"]) >= 4 else DSAT_LABEL),
                "confidence": r.get("spur_confidence", 0.5),
                "parse_ok": r.get("parse_ok", True),
            }
            for r in records
        ]
        metrics = compute_metrics(direct_like)
        print_metrics(metrics, header=f"SPUR personalized ({args.variant}) only_eval")
        _save_json(args.metrics_json, {"spur_direct": metrics})
        return

    train_samples = build_personalized_samples(
        split="train",
        train_ratio=args.train_ratio,
        seed=args.split_seed,
        min_history_sessions=args.min_history_sessions,
        target_tasks=args.target_tasks,
    )
    test_samples = build_personalized_samples(
        split="test",
        train_ratio=args.train_ratio,
        seed=args.split_seed,
        min_history_sessions=args.min_history_sessions,
        target_tasks=args.target_tasks,
    )
    train_rows = personalized_samples_to_spur_rows(train_samples, args.history_window_size)
    test_rows = personalized_samples_to_spur_rows(test_samples, args.history_window_size)

    if args.limit_train > 0:
        rng = random.Random(args.seed)
        train_rows = rng.sample(train_rows, min(args.limit_train, len(train_rows)))
    if args.limit_test > 0:
        rng = random.Random(args.seed)
        test_rows = rng.sample(test_rows, min(args.limit_test, len(test_rows)))

    logger.info(f"Train stats: {dataset_stats(train_samples)}")
    logger.info(f"Test stats:  {dataset_stats(test_samples)}")
    logger.info(f"SPUR rows: train={len(train_rows)}, test={len(test_rows)}")

    p1_cache = os.path.join(args.output_dir, "phase1_candidates.json")
    p2_cache = os.path.join(args.output_dir, f"phase2_rubrics_k{args.k_rubrics}.json")
    p3_test_cache = os.path.join(args.output_dir, f"phase3_test_{args.model}_k{args.k_rubrics}.jsonl".replace("/", "_"))

    if args.skip_phase1:
        with open(p1_cache, encoding="utf-8") as fp:
            candidates = json.load(fp)
    else:
        candidates = extract_rubric_candidates(
            train_rows,
            model=args.model,
            cache_file=p1_cache,
            max_workers=args.max_workers,
            max_per_label=args.max_extract_per_label,
        )

    if args.skip_phase2:
        with open(p2_cache, encoding="utf-8") as fp:
            rubrics = json.load(fp)
    else:
        rubrics = summarize_rubrics(
            candidates,
            model=args.model,
            k=args.k_rubrics,
            cache_file=p2_cache,
        )

    test_scored = score_rows(
        test_rows,
        rubrics=rubrics,
        model=args.model,
        cache_file=p3_test_cache,
        max_workers=args.max_workers,
        desc="SPUR personalized test",
    )

    scored_for_output = test_scored
    if args.variant != "direct":
        p3_train_cache = os.path.join(args.output_dir, f"phase3_train_{args.model}_k{args.k_rubrics}.jsonl".replace("/", "_"))
        train_scored = score_rows(
            train_rows,
            rubrics=rubrics,
            model=args.model,
            cache_file=p3_train_cache,
            max_workers=args.max_workers,
            desc="SPUR personalized train",
        )
        scored_for_output = _train_lr_variant(
            variant=args.variant,
            train_rows=train_rows,
            test_rows=test_rows,
            train_scored=train_scored,
            test_scored=test_scored,
            k_rubrics=args.k_rubrics,
            embedding_model=args.embedding_model,
            embedding_batch_size=args.embedding_batch_size,
            output_dir=args.output_dir,
            seed=args.seed,
        )

    metrics = compute_metrics(scored_for_output)
    print_metrics(metrics, header=f"SPUR personalized ({args.variant})")
    records = spur_results_to_personalized_records(
        test_rows,
        scored_for_output,
        model=args.model,
        variant=args.variant,
    )
    _save_jsonl(args.output_jsonl, records)
    _save_json(args.metrics_json, {
        f"spur_{args.variant}": metrics,
        "config": vars(args),
        "train_stats": dataset_stats(train_samples),
        "test_stats": dataset_stats(test_samples),
        "n_train_rows": len(train_rows),
        "n_test_rows": len(test_rows),
        "rubrics": rubrics,
    })
    logger.info(f"Saved compatible predictions: {args.output_jsonl}")
    logger.info(f"Saved metrics: {args.metrics_json}")


if __name__ == "__main__":
    main()
