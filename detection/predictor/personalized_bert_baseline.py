"""Supervised BERT baseline for personalized satisfaction prediction.

This baseline uses the same personalized train/test user split as
``lib.personalized_data.build_personalized_samples`` and emits JSONL records
compatible with ``eval/personalized.py``.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass

import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

_DETECTION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DETECTION_DIR not in sys.path:
    sys.path.insert(0, _DETECTION_DIR)

from lib.personalized_data import PersonalizedSample, SessionData, build_personalized_samples, dataset_stats
from lib.satisfaction_constants import normalize_reason_for_score


@dataclass
class TurnExample:
    sample_id: str
    user: str
    target_task: str
    target_file: str
    turn_idx: int
    text: str
    gold_score: int
    gold_reason: str
    source_chat_model: str


def _safe_join(values: object) -> str:
    if isinstance(values, list):
        return "，".join(str(v) for v in values)
    return str(values or "")


def format_profile(profile: dict) -> str:
    return "\n".join([
        f"性别: {profile.get('gender', '')}",
        f"年龄: {profile.get('age', '')}",
        f"背景: {profile.get('background', '')}",
        f"性格: {_safe_join(profile.get('personality', []))}",
        f"职业: {profile.get('occupation', '')}",
        f"日常兴趣: {_safe_join(profile.get('daily_interests', []))}",
        f"旅行习惯: {_safe_join(profile.get('travel_habits', []))}",
        f"饮食偏好: {_safe_join(profile.get('dining_preferences', []))}",
        f"消费习惯: {_safe_join(profile.get('spending_habits', []))}",
        f"其他方面: {_safe_join(profile.get('other_aspects', []))}",
    ])


def format_history_summary(sample: PersonalizedSample) -> str:
    scores = [
        int(score)
        for session in sample.history_sessions
        for score in session.satisfaction_scores
    ]
    if not scores:
        return "无可用历史评分。"
    dist = Counter(scores)
    mean_score = sum(scores) / len(scores)
    task_means: list[str] = []
    by_task: dict[str, list[int]] = {}
    for session in sample.history_sessions:
        by_task.setdefault(session.task, []).extend(int(s) for s in session.satisfaction_scores)
    for task, vals in sorted(by_task.items()):
        task_means.append(f"{task}: mean={sum(vals) / len(vals):.2f}, n={len(vals)}")
    return (
        f"历史轮次数: {len(scores)}\n"
        f"历史平均满意度: {mean_score:.2f}\n"
        f"历史分布: " + ", ".join(f"{i}={dist.get(i, 0)}" for i in range(1, 6)) + "\n"
        f"按任务历史: {'; '.join(task_means)}"
    )


def build_turn_text(
    sample: PersonalizedSample,
    session: SessionData,
    assistant_turn_idx: int,
    include_profile: bool,
    include_history_summary: bool,
    max_dialogue_turns: int,
) -> str:
    parts: list[str] = []
    if include_profile:
        parts.append("[USER PROFILE]\n" + format_profile(sample.profile))
    if include_history_summary:
        parts.append("[USER HISTORY SUMMARY]\n" + format_history_summary(sample))
    parts.append("[TARGET TASK]\n" + sample.target_task)
    parts.append("[TASK CONTEXT]\n" + session.task_context)

    dialogue: list[str] = []
    seen_assistant = 0
    for utt in session.history:
        role = str(utt["role"])
        content = str(utt["content"])
        dialogue.append(f"{role}: {content}")
        if role == "assistant":
            if seen_assistant == assistant_turn_idx:
                break
            seen_assistant += 1
    if max_dialogue_turns > 0:
        dialogue = dialogue[-max_dialogue_turns:]
    parts.append("[DIALOGUE]\n" + "\n".join(dialogue))
    return "\n\n".join(parts)


def build_examples(
    samples: list[PersonalizedSample],
    include_profile: bool,
    include_history_summary: bool,
    max_dialogue_turns: int,
) -> list[TurnExample]:
    examples: list[TurnExample] = []
    for sample in samples:
        for session in sample.target_sessions:
            target_file = os.path.basename(session.file_path)
            for turn_idx, score in enumerate(session.satisfaction_scores):
                reason = (
                    session.dissatisfaction_reasons[turn_idx]
                    if turn_idx < len(session.dissatisfaction_reasons)
                    else ""
                )
                gold_score = int(score)
                examples.append(TurnExample(
                    sample_id=f"{sample.user}__{sample.target_task}__{target_file}__turn_{turn_idx}",
                    user=sample.user,
                    target_task=sample.target_task,
                    target_file=target_file,
                    turn_idx=turn_idx,
                    text=build_turn_text(
                        sample=sample,
                        session=session,
                        assistant_turn_idx=turn_idx,
                        include_profile=include_profile,
                        include_history_summary=include_history_summary,
                        max_dialogue_turns=max_dialogue_turns,
                    ),
                    gold_score=gold_score,
                    gold_reason=normalize_reason_for_score(gold_score, reason),
                    source_chat_model=session.chat_model,
                ))
    return examples


class TurnDataset(Dataset):
    def __init__(self, examples: list[TurnExample], tokenizer, max_length: int, head: str):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.head = head

    def __len__(self) -> int:
        return len(self.examples)

    @staticmethod
    def _ordinal_labels(score: int) -> torch.Tensor:
        return torch.tensor([
            int(score >= 2),
            int(score >= 3),
            int(score >= 4),
            int(score >= 5),
        ], dtype=torch.float)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        encoded = self.tokenizer(
            ex.text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        if self.head == "ordinal":
            item["labels"] = self._ordinal_labels(ex.gold_score)
        else:
            item["labels"] = torch.tensor(ex.gold_score - 1, dtype=torch.long)
        return item


class BertSatisfactionModel(torch.nn.Module):
    def __init__(self, model_name: str, head: str, dropout: float):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.backbone.config.hidden_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.head_type = head
        out_dim = 4 if head == "ordinal" else 5
        self.classifier = torch.nn.Linear(hidden_size, out_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        labels = batch.get("labels")
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        outputs = self.backbone(**inputs)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(pooled))
        return logits


def logits_to_scores(
    logits: torch.Tensor,
    head: str,
    ordinal_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head == "ordinal":
        probs = torch.sigmoid(logits)
        pred = (probs > ordinal_threshold).int().sum(dim=1) + 1
        score_float = probs.sum(dim=1) + 1.0
        return pred.clamp(1, 5), score_float.clamp(1, 5)
    probs = torch.softmax(logits, dim=-1)
    pred = probs.argmax(dim=-1) + 1
    values = torch.arange(1, 6, device=logits.device, dtype=torch.float)
    score_float = (probs * values).sum(dim=-1)
    return pred.clamp(1, 5), score_float.clamp(1, 5)


def evaluate_model(
    model: BertSatisfactionModel,
    examples: list[TurnExample],
    loader: DataLoader,
    device: torch.device,
    head: str,
    ordinal_threshold: float = 0.5,
) -> tuple[dict[str, float], list[dict]]:
    model.eval()
    pred_scores: list[int] = []
    pred_float: list[float] = []
    gold_scores: list[int] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            pred, score_float = logits_to_scores(
                logits,
                head=head,
                ordinal_threshold=ordinal_threshold,
            )
            pred_scores.extend(int(v) for v in pred.cpu().tolist())
            pred_float.extend(float(v) for v in score_float.cpu().tolist())
            labels = batch["labels"]
            if head == "ordinal":
                gold = labels.sum(dim=1) + 1
            else:
                gold = labels + 1
            gold_scores.extend(int(v) for v in gold.cpu().tolist())

    metrics = compute_metrics(gold_scores, pred_scores)
    records: list[dict] = []
    for ex, pred_score, score_float in zip(examples, pred_scores, pred_float):
        records.append({
            "sample_id": ex.sample_id,
            "user": ex.user,
            "target_task": ex.target_task,
            "target_file": ex.target_file,
            "turn_idx": ex.turn_idx,
            "model": f"supervised_bert_{head}",
            "with_memory": True,
            "memory_version": "supervised_history_summary",
            "memory_update_mode": "none",
            "turn_eval_prompt_version": "supervised_bert",
            "ordinal_threshold": ordinal_threshold if head == "ordinal" else None,
            "source_chat_model": ex.source_chat_model,
            "gold_score": ex.gold_score,
            "gold_reason": ex.gold_reason,
            "pred_score": int(pred_score),
            "pred_score_float": float(score_float),
            "pred_reason": "满意" if int(pred_score) >= 4 else "其它",
            "reason_prediction": "满意" if int(pred_score) >= 4 else "其它",
            "parse_ok": True,
        })
    return metrics, records


def compute_metrics(gold: list[int], pred: list[int]) -> dict[str, float]:
    pearson = _pearson(gold, pred)
    spearman = _pearson(_rank(gold), _rank(pred))
    qwk = _quadratic_weighted_kappa(gold, pred)
    mae = sum(abs(g - p) for g, p in zip(gold, pred)) / len(gold) if gold else float("nan")
    rmse = (
        math.sqrt(sum((g - p) ** 2 for g, p in zip(gold, pred)) / len(gold))
        if gold else float("nan")
    )
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "pearson": pearson,
        "spearman": spearman,
        "qwk": qwk,
        "n": len(gold),
    }


def compute_boundary_metrics_simple(gold: list[int], pred: list[int]) -> dict[str, float]:
    gold_bin = [1 if int(v) >= 4 else 0 for v in gold]
    pred_bin = [1 if int(v) >= 4 else 0 for v in pred]
    n = len(gold_bin)
    tp_sat = sum(1 for g, p in zip(gold_bin, pred_bin) if g == 1 and p == 1)
    fp_sat = sum(1 for g, p in zip(gold_bin, pred_bin) if g == 0 and p == 1)
    fn_sat = sum(1 for g, p in zip(gold_bin, pred_bin) if g == 1 and p == 0)
    tp_dsat = sum(1 for g, p in zip(gold_bin, pred_bin) if g == 0 and p == 0)
    fp_dsat = sum(1 for g, p in zip(gold_bin, pred_bin) if g == 1 and p == 0)
    fn_dsat = sum(1 for g, p in zip(gold_bin, pred_bin) if g == 0 and p == 1)

    def _safe_div(num: float, den: float) -> float:
        return float(num / den) if den else 0.0

    def _f1(precision: float, recall: float) -> float:
        return _safe_div(2 * precision * recall, precision + recall)

    precision_sat = _safe_div(tp_sat, tp_sat + fp_sat)
    recall_sat = _safe_div(tp_sat, tp_sat + fn_sat)
    precision_dsat = _safe_div(tp_dsat, tp_dsat + fp_dsat)
    recall_dsat = _safe_div(tp_dsat, tp_dsat + fn_dsat)
    f1_sat = _f1(precision_sat, recall_sat)
    f1_dsat = _f1(precision_dsat, recall_dsat)
    n_sat_gold = sum(gold_bin)
    n_dsat_gold = n - n_sat_gold
    n_sat_pred = sum(pred_bin)
    n_dsat_pred = n - n_sat_pred
    false_sat = sum(1 for g, p in zip(gold_bin, pred_bin) if g == 0 and p == 1)
    false_dsat = sum(1 for g, p in zip(gold_bin, pred_bin) if g == 1 and p == 0)
    return {
        "n_samples": n,
        "n_sat_gold": n_sat_gold,
        "n_dsat_gold": n_dsat_gold,
        "n_sat_pred": n_sat_pred,
        "n_dsat_pred": n_dsat_pred,
        "accuracy": _safe_div(sum(1 for g, p in zip(gold_bin, pred_bin) if g == p), n),
        "f1_macro": (f1_sat + f1_dsat) / 2,
        "f1_sat": f1_sat,
        "f1_dsat": f1_dsat,
        "precision_sat": precision_sat,
        "recall_sat": recall_sat,
        "precision_dsat": precision_dsat,
        "recall_dsat": recall_dsat,
        "false_sat_rate": _safe_div(false_sat, n_dsat_gold),
        "false_dsat_rate": _safe_div(false_dsat, n_sat_gold),
    }


def _selection_value(
    metrics: dict[str, float],
    boundary_metrics: dict[str, float],
    selection_metric: str,
) -> float:
    if selection_metric == "mae":
        return -float(metrics["mae"])
    if selection_metric == "f1_dsat":
        return float(boundary_metrics["f1_dsat"])
    if selection_metric == "f1_macro":
        return float(boundary_metrics["f1_macro"])
    if selection_metric == "boundary_accuracy":
        return float(boundary_metrics["accuracy"])
    raise ValueError(f"Unsupported selection metric: {selection_metric}")


def _pearson(x: list[int] | list[float], y: list[int] | list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    mx = sum(float(v) for v in x) / len(x)
    my = sum(float(v) for v in y) / len(y)
    num = sum((float(a) - mx) * (float(b) - my) for a, b in zip(x, y))
    den_x = math.sqrt(sum((float(a) - mx) ** 2 for a in x))
    den_y = math.sqrt(sum((float(b) - my) ** 2 for b in y))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return float(num / (den_x * den_y))


def _rank(values: list[int] | list[float]) -> list[float]:
    order = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][1] == order[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k][0]] = avg_rank
        i = j + 1
    return ranks


def _quadratic_weighted_kappa(
    gold: list[int],
    pred: list[int],
    labels: list[int] | None = None,
) -> float:
    if not gold or len(gold) != len(pred):
        return float("nan")
    labels = labels or [1, 2, 3, 4, 5]
    idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    observed = [[0.0 for _ in range(n)] for _ in range(n)]
    gold_hist = [0.0 for _ in range(n)]
    pred_hist = [0.0 for _ in range(n)]
    for g, p in zip(gold, pred):
        if g not in idx or p not in idx:
            continue
        gi = idx[g]
        pi = idx[p]
        observed[gi][pi] += 1.0
        gold_hist[gi] += 1.0
        pred_hist[pi] += 1.0
    total = sum(gold_hist)
    if total == 0:
        return float("nan")
    weighted_observed = 0.0
    weighted_expected = 0.0
    denom = float((n - 1) ** 2)
    for i in range(n):
        for j in range(n):
            weight = ((i - j) ** 2) / denom
            weighted_observed += weight * observed[i][j]
            weighted_expected += weight * (gold_hist[i] * pred_hist[j] / total)
    if weighted_expected == 0:
        return 1.0 if weighted_observed == 0 else float("nan")
    return float(1.0 - weighted_observed / weighted_expected)


def split_train_valid_by_user(
    examples: list[TurnExample],
    valid_ratio: float,
    seed: int,
) -> tuple[list[TurnExample], list[TurnExample]]:
    users = sorted({ex.user for ex in examples})
    rng = random.Random(seed)
    rng.shuffle(users)
    n_valid = max(1, int(round(len(users) * valid_ratio))) if len(users) > 1 else 0
    valid_users = set(users[:n_valid])
    train = [ex for ex in examples if ex.user not in valid_users]
    valid = [ex for ex in examples if ex.user in valid_users]
    return train, valid


def save_jsonl(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Train supervised BERT personalized baseline")
    parser.add_argument("--model_name", type=str, default="bert-base-chinese")
    parser.add_argument("--head", type=str, default="ordinal", choices=["ordinal", "classification"])
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.2)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--min_history_sessions", type=int, default=1)
    parser.add_argument("--target_tasks", type=str, nargs="+", default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_dialogue_turns", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--ordinal_threshold",
        type=float,
        default=0.5,
        help="Threshold for ordinal logits at inference time; default keeps old behavior.",
    )
    parser.add_argument(
        "--dsat_weight",
        type=float,
        default=1.0,
        help="Training loss multiplier for gold_score<=3 examples.",
    )
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="mae",
        choices=["mae", "f1_dsat", "f1_macro", "boundary_accuracy"],
        help="Validation metric used to select the best checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_profile", action="store_true")
    parser.add_argument("--no_history_summary", action="store_true")
    parser.add_argument("--output_jsonl", type=str, default="")
    parser.add_argument("--metrics_json", type=str, default="")
    parser.add_argument("--checkpoint_dir", type=str, default="ckpts/supervised_bert_personalized")
    parser.add_argument("--eval_checkpoint", type=str, default="")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    include_profile = not args.no_profile
    include_history_summary = not args.no_history_summary
    train_all = build_examples(
        train_samples,
        include_profile=include_profile,
        include_history_summary=include_history_summary,
        max_dialogue_turns=args.max_dialogue_turns,
    )
    train_examples, valid_examples = split_train_valid_by_user(
        train_all,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )
    test_examples = build_examples(
        test_samples,
        include_profile=include_profile,
        include_history_summary=include_history_summary,
        max_dialogue_turns=args.max_dialogue_turns,
    )

    logger.info(f"Train stats: {dataset_stats(train_samples)}")
    logger.info(f"Test stats:  {dataset_stats(test_samples)}")
    logger.info(
        "Examples: train={}, valid={}, test={}, train_users={}, valid_users={}, test_users={}",
        len(train_examples),
        len(valid_examples),
        len(test_examples),
        len({e.user for e in train_examples}),
        len({e.user for e in valid_examples}),
        len({e.user for e in test_examples}),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = BertSatisfactionModel(args.model_name, head=args.head, dropout=args.dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(
        TurnDataset(train_examples, tokenizer, args.max_length, args.head),
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        TurnDataset(valid_examples, tokenizer, args.max_length, args.head),
        batch_size=args.eval_batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TurnDataset(test_examples, tokenizer, args.max_length, args.head),
        batch_size=args.eval_batch_size,
        shuffle=False,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_path = os.path.join(args.checkpoint_dir, "best.pt")

    if args.eval_checkpoint:
        logger.info(f"Loading checkpoint: {args.eval_checkpoint}")
        model.load_state_dict(torch.load(args.eval_checkpoint, map_location=device))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        total_steps = max(1, len(train_loader) * args.epochs)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * args.warmup_ratio),
            num_training_steps=total_steps,
        )
        loss_fn = (
            torch.nn.BCEWithLogitsLoss(reduction="none")
            if args.head == "ordinal"
            else torch.nn.CrossEntropyLoss(reduction="none")
        )
        best_selection = -math.inf
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for batch in tqdm(train_loader, desc=f"train epoch {epoch + 1}/{args.epochs}"):
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch["labels"]
                optimizer.zero_grad()
                logits = model(batch)
                if args.head == "ordinal":
                    loss_per_item = loss_fn(logits, labels.float()).mean(dim=1)
                    gold_scores = labels.sum(dim=1) + 1
                else:
                    loss_per_item = loss_fn(logits, labels)
                    gold_scores = labels + 1
                if args.dsat_weight != 1.0:
                    sample_weights = torch.where(
                        gold_scores <= 3,
                        torch.full_like(gold_scores.float(), float(args.dsat_weight)),
                        torch.ones_like(gold_scores.float()),
                    )
                    loss = (loss_per_item * sample_weights).mean()
                else:
                    loss = loss_per_item.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += float(loss.item())
            valid_metrics, valid_records = evaluate_model(
                model,
                valid_examples,
                valid_loader,
                device,
                args.head,
                ordinal_threshold=args.ordinal_threshold,
            )
            valid_boundary = compute_boundary_metrics_simple(
                [int(record["gold_score"]) for record in valid_records],
                [int(record["pred_score"]) for record in valid_records],
            )
            selection_value = _selection_value(
                valid_metrics,
                valid_boundary,
                args.selection_metric,
            )
            logger.info(
                "epoch={} loss={:.4f} valid={} valid_boundary={} selection_metric={} selection_value={:.4f}",
                epoch + 1,
                total_loss / max(1, len(train_loader)),
                valid_metrics,
                valid_boundary,
                args.selection_metric,
                selection_value,
            )
            if selection_value > best_selection:
                best_selection = selection_value
                torch.save(model.state_dict(), best_path)
                logger.info(f"Saved best checkpoint: {best_path}")
        model.load_state_dict(torch.load(best_path, map_location=device))

    test_metrics, records = evaluate_model(
        model,
        test_examples,
        test_loader,
        device,
        args.head,
        ordinal_threshold=args.ordinal_threshold,
    )
    boundary_metrics = compute_boundary_metrics_simple(
        [int(record["gold_score"]) for record in records],
        [int(record["pred_score"]) for record in records],
    )
    logger.info(f"Test metrics: {test_metrics}")
    logger.info(f"Test boundary metrics: {boundary_metrics}")

    if not args.output_jsonl:
        model_tag = args.model_name.replace("/", "_").replace(":", "_")
        args.output_jsonl = (
            f"outputs/personalized/{model_tag}_supervised_{args.head}_personalized_test.jsonl"
        )
    if not args.metrics_json:
        base, _ = os.path.splitext(args.output_jsonl)
        args.metrics_json = base + "_metrics.json"

    save_jsonl(args.output_jsonl, records)
    save_json(args.metrics_json, {
        "metrics": test_metrics,
        "boundary_metrics": boundary_metrics,
        "config": vars(args),
        "train_stats": dataset_stats(train_samples),
        "test_stats": dataset_stats(test_samples),
        "n_train_examples": len(train_examples),
        "n_valid_examples": len(valid_examples),
        "n_test_examples": len(test_examples),
    })
    logger.info(f"Saved predictions: {args.output_jsonl}")
    logger.info(f"Saved metrics: {args.metrics_json}")


if __name__ == "__main__":
    main()
