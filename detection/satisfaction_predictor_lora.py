import os
import json
import shutil
import torch
import torch.nn as nn
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
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, accuracy_score, f1_score

from metric_statistics import get_satisfaction_data
from satisfaction_predictor import format_profile
from data_split import split_by_user_group_shuffle_split
from satisfaction_constants import get_reason_to_id, get_id_to_reason
from qwen_lora_utils import get_base_model, get_model_with_lora

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

def preprocess_to_dict_data(data_list: list[dict]) -> dict[str, list[str | float | int]]:
    data = {
        'persona': [],
        'task_context': [],
        'history': [],
        'assistant_reply': [],
        'score': [],
        'reason': [],
        'file_path': [],
        'turn': [],
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
                data['file_path'].append(sample['file_path'])
                data['turn'].append(assistant_turn_idx + 1)  # 1-based 对话轮次
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

def run_test_only(
    model_name: str,
    max_len: int,
    batch_size: int,
    checkpoint_path: str,
    test_output: str,
):
    """仅加载 checkpoint 在测试集上预测，并保存预测结果、labels、数据文件与对话轮次供单独统计。"""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {DEVICE}, test_only mode")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dataset, num_reasons, reason_to_id = get_dataset(tokenizer, max_len)
    train_idx, valid_idx, test_idx = split_by_user_group_shuffle_split(dataset["user"], train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    id_to_reason = get_id_to_reason()
    test_dataset = dataset.select(test_idx)
    logger.info(f"Test samples: {len(test_idx)}")

    base_model = get_base_model(model_name)
    backbone = get_model_with_lora(base_model)
    model = SatisfactionModel(backbone, num_reason_classes=num_reasons)

    # 加载 checkpoint（兼容 safetensors / pytorch_model.bin）
    ckpt_model_path = None
    for name in ("model.safetensors", "pytorch_model.bin"):
        p = os.path.join(checkpoint_path, name)
        if os.path.isfile(p):
            ckpt_model_path = p
            break
    if ckpt_model_path is None:
        raise FileNotFoundError(f"No model weights found in {checkpoint_path} (need model.safetensors or pytorch_model.bin)")

    if ckpt_model_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(ckpt_model_path, device="cpu")
    else:
        state_dict = torch.load(ckpt_model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)

    training_args = TrainingArguments(
        output_dir=os.path.join(checkpoint_path, "eval_tmp"),
        per_device_eval_batch_size=batch_size,
        bf16=True,
    )
    trainer = Trainer(model=model, args=training_args)
    pred_output = trainer.predict(test_dataset)

    pred_scores = pred_output.predictions[0].tolist()
    reason_logits = pred_output.predictions[1]
    pred_reason_ids = reason_logits.argmax(axis=-1).tolist()
    true_scores = pred_output.label_ids[0].tolist()
    true_reason_ids = pred_output.label_ids[1].tolist()

    results = []
    for i in range(len(test_dataset)):
        row = test_dataset[i]
        results.append({
            "file_path": row["file_path"],
            "turn": row["turn"],
            "label_score": float(true_scores[i]),
            "label_reason": id_to_reason[int(true_reason_ids[i])],
            "pred_score": round(float(pred_scores[i]), 4),
            "pred_reason": id_to_reason[int(pred_reason_ids[i])],
        })

    os.makedirs(os.path.dirname(test_output) or ".", exist_ok=True)
    with open(test_output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Test predictions and labels saved to {test_output} (total {len(results)} rows)")
    if os.path.isdir(training_args.output_dir):
        shutil.rmtree(training_args.output_dir, ignore_errors=True)
    return results


def main(model_name: str = "Qwen/Qwen3-8B", max_len: int = 1024, batch_size: int = 2, eval_batch_size: int = 8, num_epochs: int = 10, resume_from_checkpoint: bool = False, test_only: bool = False, checkpoint_path: str = "", test_output: str = "test_predictions.json"):
    if test_only:
        if not checkpoint_path:
            raise ValueError("--test_only 时必须指定 --checkpoint_path")
        run_test_only(model_name=model_name, max_len=max_len, batch_size=eval_batch_size, checkpoint_path=checkpoint_path, test_output=test_output)
        return

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {DEVICE}")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dataset, num_reasons, reason_to_id = get_dataset(tokenizer, max_len)
    train_idx, valid_idx, test_idx = split_by_user_group_shuffle_split(dataset["user"], train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
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
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument("--test_only", action="store_true", help="仅加载已有 checkpoint 做测试，并保存预测与标签到文件")
    parser.add_argument("--checkpoint_path", type=str, default="", help="test_only 时指定 checkpoint 目录（含 model.safetensors 或 pytorch_model.bin）")
    parser.add_argument("--test_output", type=str, default="output/test_predictions.json", help="test_only 时保存结果的 JSON 路径")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
