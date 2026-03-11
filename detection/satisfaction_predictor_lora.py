import torch
import torch.nn as nn
from functools import partial
from loguru import logger
from datasets import Dataset
from argparse import ArgumentParser
from transformers import (
    AutoTokenizer,
    AutoModel,
    PreTrainedTokenizer,
    PreTrainedModel,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from peft import LoraConfig, get_peft_model
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, accuracy_score, f1_score

from metric_statistics import get_satisfaction_data
from satisfaction_predictor import format_profile

# =========================
# 添加回归 Head
# =========================

class SatisfactionModel(nn.Module):
    def __init__(self, backbone: PreTrainedModel, num_reason_classes: int):
        super().__init__()
        self.backbone = backbone
        hidden_size = backbone.config.hidden_size
        self.regressor = nn.Linear(hidden_size, 1)
        self.reason_classifier = nn.Linear(hidden_size, num_reason_classes)
        self.satisfaction_loss_fn = nn.MSELoss()
        self.reason_loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor | None = None, reason_labels: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
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
        score = self.regressor(mean_hidden).squeeze(-1)
        reason_logits = self.reason_classifier(mean_hidden)

        loss = None
        if labels is not None:
            loss = self.satisfaction_loss_fn(score, labels.float())
            if reason_labels is not None:
                loss += self.reason_loss_fn(reason_logits, reason_labels)

        return {"loss": loss, "logits": score, "reason_logits": reason_logits}

def preprocess_to_dict_data(data_list: list[dict]) -> dict[str, list[str | float]]:
    data = {
        'persona': [],
        'task_context': [],
        'history': [],
        'assistant_reply': [],
        'score': [],
        'reason': [],
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
                assistant_turn_idx += 1
            history.append(f"{utt['role']}：{utt['content']}\n")
            while len(history) > 5:  # 只保留最近的对话历史，防止过长
                history.pop(0)
    return data

# =========================
# Tokenizer
# =========================

def format_example(example: dict[str, str | float]) -> str:
    text = (
        f"用户画像：{example['persona']}\n\n"
        f"任务背景：{example['task_context']}\n\n"
        f"最近对话历史：{example['history']}\n\n"
        f"当前助手回复：{example['assistant_reply']}\n\n"
        f"请预测用户满意度（1-5分）："
    )
    return text

def tokenize_function(example: dict[str, str | float], tokenizer: PreTrainedTokenizer, max_len: int, reason_to_id: dict[str, int]) -> dict[str, torch.Tensor | float | int]:
    text = format_example(example)
    tokenized = tokenizer(
        text,
        max_length=max_len,
        truncation=True,
        padding="max_length"
    )
    tokenized["labels"] = example["score"]
    tokenized["reason_labels"] = reason_to_id[example["reason"]]
    return tokenized

# =========================
# 数据准备
# =========================

def get_dataset(tokenizer: PreTrainedTokenizer, max_len: int) -> tuple[Dataset, int, dict]:
    data_list = get_satisfaction_data()  # 从 metric_statistics 获取数据
    data = preprocess_to_dict_data(data_list)
    reason_to_id = {'其它': 0, '不够多样': 1, '不可用': 2, '满意': 3, '不够细致': 4, '不满足需求': 5}
    num_reasons = len(reason_to_id)
    logger.info(reason_to_id)

    dataset = Dataset.from_dict(data)
    partial_tokenize = partial(tokenize_function, tokenizer=tokenizer, max_len=max_len, reason_to_id=reason_to_id)
    dataset = dataset.map(partial_tokenize)
    return dataset, num_reasons, reason_to_id

# =========================
# 加载 Qwen3 backbone（仅 encoder 使用）
# =========================

def get_base_model(model_name: str) -> PreTrainedModel:
    base_model = AutoModel.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    return base_model

# =========================
# 加 LoRA
# =========================

def get_model_with_lora(base_model: PreTrainedModel) -> PreTrainedModel:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # Qwen常用
        lora_dropout=0.1,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    peft_model = get_peft_model(base_model, lora_config)
    return peft_model

# =========================
# Trainer 配置
# =========================

def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    pred_scores, reason_logits = eval_pred.predictions
    true_scores, true_reasons = eval_pred.label_ids
    mae = mean_absolute_error(true_scores, pred_scores)
    rmse = root_mean_squared_error(true_scores, pred_scores)
    r2 = r2_score(true_scores, pred_scores)
    pearson_corr = pearsonr(true_scores, pred_scores)[0]
    spearman_corr = spearmanr(true_scores, pred_scores)[0]
    reason_preds = reason_logits.argmax(axis=-1)
    accuracy = accuracy_score(true_reasons, reason_preds)
    f1 = f1_score(true_reasons, reason_preds, average="weighted")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
        "reason_accuracy": accuracy,
        "reason_f1": f1,
    }

def get_trainer(model: nn.Module, train_dataset: Dataset, valid_dataset: Dataset, batch_size: int, num_epochs: int) -> Trainer:
    training_args = TrainingArguments(
        output_dir="./ckpts/llm_predictor",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
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
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.1, random_state=42)
    train_idx, valid_idx = train_test_split(train_idx, test_size=1.0 / 9.0, random_state=42)
    logger.info(f"Train samples: {len(train_idx)}, Valid samples: {len(valid_idx)}, Test samples: {len(test_idx)}")
    base_model = get_base_model(model_name)
    backbone = get_model_with_lora(base_model)
    model = SatisfactionModel(backbone, num_reason_classes=num_reasons)
    model.to(DEVICE)
    train_dataset = dataset.select(train_idx)
    valid_dataset = dataset.select(valid_idx)
    test_dataset = dataset.select(test_idx)
    trainer = get_trainer(model, train_dataset, valid_dataset, batch_size, num_epochs)
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
