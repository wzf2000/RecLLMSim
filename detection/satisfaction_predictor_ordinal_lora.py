import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from loguru import logger
from datasets import Dataset
from argparse import ArgumentParser
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, accuracy_score, f1_score, cohen_kappa_score

from metric_statistics import get_satisfaction_data
from satisfaction_predictor import format_profile
from data_split import split_by_user_group_shuffle_split
from satisfaction_constants import get_reason_to_id
from qwen_lora_utils import get_base_model, get_model_with_lora

# =========================
# 添加 Ordinal Head
# =========================

class SatisfactionModel(nn.Module):
    def __init__(
        self,
        backbone: PreTrainedModel,
        num_reason_classes: int,
        satisfied_reason_id: int,
        score_weights: list[float] | None = None,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.1,
        delta: float = 0.2,
        consistency_temp: float = 2.0,
        consistency_center: float = 3.5,
    ):
        super().__init__()
        self.backbone = backbone
        hidden_size = backbone.config.hidden_size
        self.ordinal_head = torch.nn.Linear(hidden_size, 4)  # For score ordinal regression (2, 3, 4, 5)
        self.reason_classifier = nn.Linear(hidden_size, num_reason_classes)
        self.satisfaction_loss_fn = nn.BCEWithLogitsLoss()
        self.reason_loss_fn = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.consistency_temp = consistency_temp
        self.consistency_center = consistency_center
        self.satisfied_reason_id = satisfied_reason_id
        if score_weights is not None:
            self.register_buffer("score_weights", torch.tensor(score_weights, dtype=torch.float))
        else:
            self.score_weights = None

    def monotonic_penalty(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        diff = probs[:, 1:] - probs[:, :-1]
        penalty = torch.relu(diff).mean()
        return penalty

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        reason_labels: torch.Tensor | None = None,
        score_int: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 做 mean pooling
        hidden = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        sum_hidden = torch.sum(hidden * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        mean_hidden = sum_hidden / sum_mask
        ordinal_logtis = self.ordinal_head(mean_hidden)
        reason_logits = self.reason_classifier(mean_hidden)

        loss = None
        if labels is not None:
            # 对 ordinal 满意度损失做“按分数加权”，重点提升低分（尤其 1 分）学习信号
            per_dim = F.binary_cross_entropy_with_logits(ordinal_logtis, labels.float(), reduction="none")  # (bs, 4)
            per_sample = per_dim.mean(dim=1)  # (bs,)
            if score_int is not None and self.score_weights is not None:
                w = self.score_weights[score_int.long()].to(per_sample.dtype)
                per_sample = per_sample * w
            loss = self.alpha * per_sample.mean()
            loss += self.gamma * self.monotonic_penalty(ordinal_logtis)
            if reason_labels is not None:
                loss += self.beta * self.reason_loss_fn(reason_logits, reason_labels)

            # 跨任务一致性约束：
            # 分数越高 => "满意" 概率越高；分数越低 => "满意" 概率越低
            # 用预测的 ordinal logits 得到期望分数，再映射为满意概率目标。
            if self.delta > 0:
                # expected_score in [1, 5]
                expected_score = 1.0 + torch.sigmoid(ordinal_logtis).sum(dim=1)
                target_satisfied_prob = torch.sigmoid(
                    (expected_score - self.consistency_center) * self.consistency_temp
                )
                # 使用 "满意" 的 softmax-logit 做 BCEWithLogits（autocast 安全）
                # p = softmax(z)[k] => logit(p) = z_k - logsumexp(z)
                satisfied_softmax_logit = reason_logits[:, self.satisfied_reason_id] - torch.logsumexp(reason_logits, dim=-1)
                consistency_loss = F.binary_cross_entropy_with_logits(satisfied_softmax_logit, target_satisfied_prob)
                loss += self.delta * consistency_loss

        return {"loss": loss, "logits": ordinal_logtis, "reason_logits": reason_logits}

def preprocess_to_dict_data(data_list: list[dict]) -> dict[str, list[str | float]]:
    data = {
        'persona': [],
        'task_context': [],
        'history': [],
        'assistant_reply': [],
        'score': [],
        'reason': [],
        'user': [],
    }
    for sample in data_list:
        persona = format_profile(sample['profile'])
        task_context = sample['task_context']
        history = []
        assistant_turn_idx = 0
        for utt in sample['history']:
            if utt['role'] == 'assistant':
                data['persona'].append(persona)
                data['task_context'].append(task_context)
                data['history'].append("\n".join(history))
                data['assistant_reply'].append(utt['content'])
                data['score'].append(sample['satisfaction_scores'][assistant_turn_idx])
                data['reason'].append(sample['dissatisfaction_reasons'][assistant_turn_idx])
                data['user'].append(sample.get('user', 'unknown'))
                assistant_turn_idx += 1
            history.append(f"{utt['role']}：{utt['content']}\n")
            while len(history) > 5:  # 只保留最近的对话历史，防止过长
                history.pop(0)
    return data


# =========================
# Tokenizer
# =========================

def format_example(example: dict[str, str | float]) -> str:
    persona = (example.get("persona") or "").strip()
    task_context = (example.get("task_context") or "").strip()
    history = (example.get("history") or "").strip()
    assistant_reply = (example.get("assistant_reply") or "").strip()

    text = (
        f"用户画像：{persona}\n\n"
        f"任务背景：{task_context}\n\n"
        f"最近对话历史：{history}\n\n"
        f"当前助手回复：{assistant_reply}\n\n"
        f"请预测用户满意度（1-5分）："
    )
    return text

def score_to_ordinal(score: int) -> list[int]:
    return [
        int(score >= 2),
        int(score >= 3),
        int(score >= 4),
        int(score >= 5),
    ]

def tokenize_function(example: dict[str, str | float], tokenizer: PreTrainedTokenizer, max_len: int, reason_to_id: dict[str, int]) -> dict[str, torch.Tensor | float | int]:
    text = format_example(example)
    tokenized = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        padding="max_length"
    )
    tokenized["labels"] = torch.tensor(score_to_ordinal(example["score"]), dtype=torch.float)
    tokenized["score_int"] = int(example["score"])
    tokenized["reason_labels"] = reason_to_id[example["reason"]]
    return tokenized

# =========================
# 数据准备
# =========================

def get_dataset(tokenizer: PreTrainedTokenizer, max_len: int) -> tuple[Dataset, int, dict]:
    data_list = get_satisfaction_data()  # 从 metric_statistics 获取数据
    data = preprocess_to_dict_data(data_list)
    reason_to_id = get_reason_to_id()
    num_reasons = len(reason_to_id)
    logger.info(reason_to_id)

    dataset = Dataset.from_dict(data)
    partial_tokenize = partial(tokenize_function, tokenizer=tokenizer, max_len=max_len, reason_to_id=reason_to_id)
    dataset = dataset.map(partial_tokenize)
    return dataset, num_reasons, reason_to_id

# =========================
# Trainer 配置
# =========================

def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    ordinal_logits, reason_logits = eval_pred.predictions
    ordinal_pred = (ordinal_logits > 0).astype(int)  # (batch_size, 4)
    pred_scores = ordinal_pred.sum(axis=1) + 1
    ordinal_labels, true_reasons = eval_pred.label_ids
    true_scores = ordinal_labels.sum(axis=1) + 1
    mae = mean_absolute_error(true_scores, pred_scores)
    rmse = root_mean_squared_error(true_scores, pred_scores)
    r2 = r2_score(true_scores, pred_scores)
    kappa = cohen_kappa_score(true_scores, pred_scores, weights='quadratic')
    pearson_corr = pearsonr(true_scores, pred_scores)[0]
    spearman_corr = spearmanr(true_scores, pred_scores)[0]
    reason_preds = reason_logits.argmax(axis=-1)
    accuracy = accuracy_score(true_reasons, reason_preds)
    f1 = f1_score(true_reasons, reason_preds, average="weighted")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "kappa": kappa,
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
        "reason_accuracy": accuracy,
        "reason_f1": f1,
    }

def get_trainer(model: nn.Module, train_dataset: Dataset, valid_dataset: Dataset, batch_size: int, num_epochs: int) -> Trainer:
    training_args = TrainingArguments(
        output_dir="./ckpts/llm_predictor_ordinal",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        num_train_epochs=num_epochs,
        learning_rate=2e-4,
        optim="adamw_torch",
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="mae",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )
    return trainer

def main(model_name: str = "Qwen/Qwen3-8B", max_len: int = 1024, batch_size: int = 2, num_epochs: int = 10, resume_from_checkpoint: bool = False):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {DEVICE}")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dataset, num_reasons, reason_to_id = get_dataset(tokenizer, max_len)
    train_idx, valid_idx, test_idx = split_by_user_group_shuffle_split(dataset["user"], train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    logger.info(f"Train samples: {len(train_idx)}, Valid samples: {len(valid_idx)}, Test samples: {len(test_idx)}")

    # 用训练集分布构建分数权重（逆频率），缓解 1 分样本学习不足问题
    train_scores = [int(s) for s in dataset.select(train_idx)["score"]]
    total = len(train_scores)
    counts = {i: 0 for i in range(1, 6)}
    for s in train_scores:
        if s in counts:
            counts[s] += 1
    score_weights = [0.0] * 6  # index 0 unused
    for s in range(1, 6):
        c = counts[s]
        if c <= 0:
            score_weights[s] = 1.0
        else:
            score_weights[s] = total / (5.0 * c)
    # 防止极端权重导致训练不稳定
    score_weights = [min(5.0, max(0.2, w)) for w in score_weights]
    logger.info(f"Train score counts: {counts}, score_weights: {score_weights[1:]}")

    base_model = get_base_model(model_name)
    backbone = get_model_with_lora(base_model)
    model = SatisfactionModel(
        backbone,
        num_reason_classes=num_reasons,
        satisfied_reason_id=reason_to_id["满意"],
        score_weights=score_weights,
    )
    model.to(DEVICE)
    train_dataset = dataset.select(train_idx)
    valid_dataset = dataset.select(valid_idx)
    test_dataset = dataset.select(test_idx)
    trainer = get_trainer(model, train_dataset, valid_dataset, batch_size, num_epochs)
    init_metrics = trainer.evaluate()  # 先评估一下初始模型性能
    logger.info(f"Initial Metrics: {init_metrics}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    test_metrics = trainer.predict(test_dataset).metrics
    logger.info(f"Test Metrics: {test_metrics}")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
