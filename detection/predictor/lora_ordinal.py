import os
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

from lib.metric_statistics import get_satisfaction_data
from lib.uss_data import get_uss_flat_data, ALL_DATASETS as USS_ALL_DATASETS
from lib.data_split import split_by_user_group_shuffle_split
from lib.satisfaction_constants import get_reason_to_id
from lib.qwen_lora_utils import get_base_model, get_model_with_lora
from predictor.bert import format_profile

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
        reason_class_weights: list[float] | None = None,
        alpha: float = 1.0,
        beta: float = 2.0,
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
        if reason_class_weights is not None:
            self.register_buffer("reason_weight", torch.tensor(reason_class_weights, dtype=torch.float))
        else:
            self.reason_weight = None

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
                if self.reason_weight is not None:
                    reason_loss = F.cross_entropy(reason_logits, reason_labels,
                                                  weight=self.reason_weight.to(reason_logits.dtype))
                else:
                    reason_loss = self.reason_loss_fn(reason_logits, reason_labels)
                loss += self.beta * reason_loss

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

def tokenize_function(
    example: dict[str, str | float],
    tokenizer: PreTrainedTokenizer,
    max_len: int,
    reason_to_id: dict[str, int],
    disable_reason: bool = False,
) -> dict[str, torch.Tensor | float | int]:
    text = format_example(example)
    tokenized = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        padding="max_length"
    )
    tokenized["labels"] = torch.tensor(score_to_ordinal(example["score"]), dtype=torch.float)
    tokenized["score_int"] = int(example["score"])
    if not disable_reason:
        tokenized["reason_labels"] = reason_to_id[example["reason"]]
    return tokenized

# =========================
# 数据准备
# =========================

def get_dataset(
    tokenizer: PreTrainedTokenizer,
    max_len: int,
    data_list: list[dict] | None = None,
    disable_reason: bool = False,
    is_flat: bool = False,
) -> tuple[Dataset, int, dict]:
    if data_list is None:
        data_list = get_satisfaction_data()
    if is_flat:
        # data_list is already turn-level dicts; transpose to dict-of-lists
        data = {k: [d[k] for d in data_list] for k in data_list[0]}
    else:
        data = preprocess_to_dict_data(data_list)
    reason_to_id = get_reason_to_id()
    num_reasons = len(reason_to_id)
    logger.info(f"reason_to_id={reason_to_id}, disable_reason={disable_reason}")

    dataset = Dataset.from_dict(data)
    partial_tokenize = partial(
        tokenize_function,
        tokenizer=tokenizer, max_len=max_len,
        reason_to_id=reason_to_id, disable_reason=disable_reason,
    )
    dataset = dataset.map(partial_tokenize)
    return dataset, num_reasons, reason_to_id

# =========================
# Trainer 配置
# =========================

def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    ordinal_logits, reason_logits = eval_pred.predictions
    ordinal_pred = (ordinal_logits > 0).astype(int)  # (batch_size, 4)
    pred_scores = ordinal_pred.sum(axis=1) + 1
    # label_ids 为 tuple 时含 reason_labels，否则仅有 ordinal labels（disable_reason 模式）
    if isinstance(eval_pred.label_ids, tuple):
        ordinal_labels, true_reasons = eval_pred.label_ids
        has_reason = True
    else:
        ordinal_labels = eval_pred.label_ids
        has_reason = False
    true_scores = ordinal_labels.sum(axis=1) + 1
    mae = mean_absolute_error(true_scores, pred_scores)
    rmse = root_mean_squared_error(true_scores, pred_scores)
    r2 = r2_score(true_scores, pred_scores)
    kappa = cohen_kappa_score(true_scores, pred_scores, weights='quadratic')
    score_accuracy = accuracy_score(true_scores, pred_scores)
    pearson_corr = pearsonr(true_scores, pred_scores)[0]
    spearman_corr = spearmanr(true_scores, pred_scores)[0]
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "kappa": kappa,
        "score_accuracy": score_accuracy,
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
    }
    if has_reason:
        reason_preds = reason_logits.argmax(axis=-1)
        metrics["reason_accuracy"] = accuracy_score(true_reasons, reason_preds)
        metrics["reason_f1"] = f1_score(true_reasons, reason_preds, average="weighted", zero_division=0)
    return metrics

def get_trainer(
    model: nn.Module,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    batch_size: int,
    num_epochs: int,
    output_dir: str,
) -> Trainer:
    training_args = TrainingArguments(
        output_dir=output_dir,
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

def run_test_only(
    model_name: str,
    max_len: int,
    batch_size: int,
    checkpoint_path: str,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.1,
    delta: float = 0.2,
    consistency_temp: float = 2.0,
    consistency_center: float = 3.5,
    data_list: list[dict] | None = None,
    disable_reason: bool = False,
    is_flat: bool = False,
    sft_model_path: str = "",
):
    """加载已有 checkpoint，在测试集上评测并输出统一指标。"""
    import shutil
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {DEVICE}, test_only mode")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dataset, num_reasons, reason_to_id = get_dataset(tokenizer, max_len, data_list=data_list, disable_reason=disable_reason, is_flat=is_flat)
    train_idx, valid_idx, test_idx = split_by_user_group_shuffle_split(
        dataset["user"], train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
    )
    test_dataset = dataset.select(test_idx)
    logger.info(f"Test samples: {len(test_idx)}")

    backbone_name = sft_model_path if sft_model_path else model_name
    if sft_model_path:
        logger.info(f"Using SFT model as backbone: {sft_model_path}")
    base_model = get_base_model(backbone_name)
    backbone = get_model_with_lora(base_model)
    model = SatisfactionModel(
        backbone,
        num_reason_classes=num_reasons,
        satisfied_reason_id=reason_to_id["满意"],
        alpha=alpha, beta=beta, gamma=gamma, delta=delta,
        consistency_temp=consistency_temp, consistency_center=consistency_center,
    )

    # 加载 checkpoint（兼容 safetensors / pytorch_model.bin）
    ckpt_model_path = None
    for name in ("model.safetensors", "pytorch_model.bin"):
        p = os.path.join(checkpoint_path, name)
        if os.path.isfile(p):
            ckpt_model_path = p
            break
    if ckpt_model_path is None:
        raise FileNotFoundError(
            f"No model weights found in {checkpoint_path} (need model.safetensors or pytorch_model.bin)"
        )
    if ckpt_model_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_model_path, device="cpu")
    else:
        state_dict = torch.load(ckpt_model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)

    eval_tmp_dir = os.path.join(checkpoint_path, "eval_tmp")
    training_args = TrainingArguments(
        output_dir=eval_tmp_dir,
        per_device_eval_batch_size=batch_size,
        bf16=True,
        fp16=False,
    )
    trainer = Trainer(model=model, args=training_args, compute_metrics=compute_metrics)
    pred_output = trainer.predict(test_dataset)
    logger.info(f"Test Metrics: {pred_output.metrics}")
    if os.path.isdir(eval_tmp_dir):
        shutil.rmtree(eval_tmp_dir, ignore_errors=True)


def _load_data_list(
    data_source: str,
    uss_datasets: list[str] | None,
    uss_data_dir: str,
) -> tuple[list[dict], bool]:
    """根据 data_source 加载数据。返回 (data_list, is_flat)。
    is_flat=True 表示 data_list 已是 turn-level 平铺 dicts，无需再经 preprocess_to_dict_data()。
    """
    if data_source == "uss":
        flat = get_uss_flat_data(
            datasets=uss_datasets,
            data_dir=uss_data_dir,
            splits=None,   # 加载全部，由 GroupShuffleSplit 重新划分
        )
        return flat, True
    else:
        return get_satisfaction_data(), False


def main(
    model_name: str = "Qwen/Qwen3-8B",
    max_len: int = 1024,
    batch_size: int = 2,
    num_epochs: int = 10,
    resume_from_checkpoint: bool = False,
    output_dir: str = "./ckpts/llm_predictor_ordinal",
    alpha: float = 1.0,
    beta: float = 2.0,
    gamma: float = 0.1,
    delta: float = 0.2,
    consistency_temp: float = 2.0,
    consistency_center: float = 3.5,
    use_score_weights: bool = False,
    use_reason_weights: bool = False,
    test_only: bool = False,
    checkpoint_path: str = "",
    data_source: str = "internal",
    uss_datasets: list[str] | None = None,
    uss_data_dir: str = "./data/uss/processed",
    disable_reason: bool = False,
    sft_model_path: str = "",
):
    data_list, is_flat = _load_data_list(data_source, uss_datasets, uss_data_dir)

    if test_only:
        if not checkpoint_path:
            raise ValueError("--test_only 时必须指定 --checkpoint_path")
        run_test_only(
            model_name=model_name, max_len=max_len, batch_size=batch_size,
            checkpoint_path=checkpoint_path,
            alpha=alpha, beta=beta, gamma=gamma, delta=delta,
            consistency_temp=consistency_temp, consistency_center=consistency_center,
            data_list=data_list, disable_reason=disable_reason, is_flat=is_flat,
            sft_model_path=sft_model_path,
        )
        return

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {DEVICE}, data_source={data_source}, disable_reason={disable_reason}")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dataset, num_reasons, reason_to_id = get_dataset(tokenizer, max_len, data_list=data_list, disable_reason=disable_reason, is_flat=is_flat)
    train_idx, valid_idx, test_idx = split_by_user_group_shuffle_split(dataset["user"], train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    logger.info(f"Train samples: {len(train_idx)}, Valid samples: {len(valid_idx)}, Test samples: {len(test_idx)}")

    score_weights: list[float] | None = None
    if use_score_weights:
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
        score_weights = [min(8.0, max(0.2, w)) for w in score_weights]
        logger.info(f"Train score counts: {counts}, score_weights: {score_weights[1:]}")
    else:
        logger.info("Disable score_weights (use_score_weights=0)")

    reason_class_weights: list[float] | None = None
    if use_reason_weights and not disable_reason:
        train_reasons = [r for r in dataset.select(train_idx)["reason"]]
        total_r = len(train_reasons)
        reason_ids = list(reason_to_id.values())
        id_to_reason = {v: k for k, v in reason_to_id.items()}
        reason_counts = {i: 0 for i in range(len(reason_to_id))}
        for r in train_reasons:
            rid = reason_to_id.get(r)
            if rid is not None:
                reason_counts[rid] += 1
        reason_class_weights = []
        for i in range(len(reason_to_id)):
            c = reason_counts[i]
            if c <= 0:
                reason_class_weights.append(1.0)
            else:
                reason_class_weights.append(total_r / (len(reason_to_id) * c))
        reason_class_weights = [min(8.0, max(0.2, w)) for w in reason_class_weights]
        logger.info(f"Reason class weights: { {id_to_reason[i]: round(reason_class_weights[i], 3) for i in range(len(reason_to_id))} }")
    else:
        logger.info("Disable reason_class_weights (use_reason_weights=0)")

    backbone_name = sft_model_path if sft_model_path else model_name
    if sft_model_path:
        logger.info(f"Using SFT model as backbone: {sft_model_path}")
    base_model = get_base_model(backbone_name)
    backbone = get_model_with_lora(base_model)
    model = SatisfactionModel(
        backbone,
        num_reason_classes=num_reasons,
        satisfied_reason_id=reason_to_id["满意"],
        score_weights=score_weights,
        reason_class_weights=reason_class_weights,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        consistency_temp=consistency_temp,
        consistency_center=consistency_center,
    )
    model.to(DEVICE)
    train_dataset = dataset.select(train_idx)
    valid_dataset = dataset.select(valid_idx)
    test_dataset = dataset.select(test_idx)
    trainer = get_trainer(model, train_dataset, valid_dataset, batch_size, num_epochs, output_dir=output_dir)
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
    parser.add_argument("--output_dir", type=str, default="./ckpts/llm_predictor_ordinal")
    parser.add_argument("--alpha", type=float, default=1.0, help="ordinal satisfaction loss 系数")
    parser.add_argument("--beta", type=float, default=2.0, help="reason classification loss 系数")
    parser.add_argument("--gamma", type=float, default=0.1, help="monotonic_penalty 系数")
    parser.add_argument("--delta", type=float, default=0.2, help="跨任务一致性约束 loss 系数；=0 表示禁用")
    parser.add_argument("--consistency_temp", type=float, default=2.0, help="一致性约束的 sigmoid 温度")
    parser.add_argument("--consistency_center", type=float, default=3.5, help="一致性约束的分数中心")
    parser.add_argument(
        "--use_score_weights",
        action="store_true",
        help="是否使用按分数加权",
    )
    parser.add_argument(
        "--use_reason_weights",
        action="store_true",
        help="是否使用 reason 类别逆频率权重（缓解 满意 类过多的不平衡问题）",
    )
    parser.add_argument("--test_only", action="store_true", help="仅加载已有 checkpoint 做测试，跳过训练")
    parser.add_argument("--checkpoint_path", type=str, default="", help="test_only 时指定 checkpoint 目录（含 model.safetensors 或 pytorch_model.bin）")
    parser.add_argument(
        "--data_source", type=str, default="internal", choices=["internal", "uss"],
        help="训练数据来源：internal=项目内部数据，uss=USS 公开数据集",
    )
    parser.add_argument(
        "--uss_datasets", nargs="+", default=None, choices=USS_ALL_DATASETS,
        help="USS 模式下使用的子数据集，默认全部（JDDC/SGD/MWOZ/ReDial/CCPE）",
    )
    parser.add_argument(
        "--uss_data_dir", type=str, default="./data/uss/processed",
        help="USS 预处理 JSONL 目录（tools/preprocess_uss.py 输出目录）",
    )
    parser.add_argument(
        "--disable_reason", action="store_true",
        help="禁用 reason 分类器和相关 loss（USS 等无细粒度原因标注的数据集使用）",
    )
    parser.add_argument(
        "--sft_model_path", type=str, default="",
        help="用自己训练的 SFT 模型替换 base model 作为 backbone（传入 checkpoint 目录）；为空则使用 --model_name",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
