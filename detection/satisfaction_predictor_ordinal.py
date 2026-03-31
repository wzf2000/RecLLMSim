import os
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from loguru import logger
from argparse import ArgumentParser
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, accuracy_score, f1_score, cohen_kappa_score
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizer
from transformers.utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy

from metric_statistics import get_satisfaction_data
from data_split import split_by_user_group_shuffle_split
from satisfaction_constants import get_reason_to_id

class OrdinalHead(torch.nn.Module):
    def __init__(self, hidden_size: int, num_classes: int = 5):
        super().__init__()
        self.fc = torch.nn.Linear(hidden_size, 1, bias=False)
        self.bias = torch.nn.Parameter(
            torch.zeros(num_classes - 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.fc(x)
        logits = logits - self.bias
        return logits

class SatisfactionPredictor(torch.nn.Module):
    def __init__(self, backbone: PreTrainedModel, num_reasons: int):
        super(SatisfactionPredictor, self).__init__()
        self.backbone = backbone
        hidden_size = backbone.config.hidden_size
        # self.ordinal_head = OrdinalHead(hidden_size, 5)  # For score ordinal regression (2, 3, 4, 5)
        self.ordinal_head = torch.nn.Linear(hidden_size, 4)  # For score ordinal regression (2, 3, 4, 5)
        self.classification_head = torch.nn.Linear(hidden_size, num_reasons)  # For reason classification

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        ordinal_logits = self.ordinal_head(pooled_output)  # [batch_size, 4]
        reason_logits = self.classification_head(pooled_output)  # [batch_size, num_reasons]
        return ordinal_logits, reason_logits

class SatisfactionDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], reasons: list[str], tokenizer: PreTrainedTokenizer, reason_to_id: dict[str, int]):
        self.texts = texts
        self.labels = labels
        self.reasons = reasons
        self.tokenizer = tokenizer
        self.reason_to_id = reason_to_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.max_length = tokenizer.model_max_length

    def __len__(self):
        return len(self.texts)

    def score_to_ordinal(self, score: int) -> list[int]:
        return [
            int(score >= 2),
            int(score >= 3),
            int(score >= 4),
            int(score >= 5),
        ]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        reason = self.reasons[idx]
        reason_id = self.reason_to_id[reason]

        encoding = self.tokenizer._encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding_strategy=PaddingStrategy.MAX_LENGTH,
            truncation_strategy=TruncationStrategy.LONGEST_FIRST,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.score_to_ordinal(label), dtype=torch.float),
            'reasons': reason_id
        }

def format_profile(profile: dict) -> str:
    profile_str = f"性别: {profile['gender']}\n年龄: {profile['age']}\n背景: {profile['background']}\n性格: {', '.join(profile['personality'])}\n职业: {profile['occupation']}\n日常兴趣: {', '.join(profile['daily_interests'])}\n旅行习惯: {', '.join(profile['travel_habits'])}\n饮食偏好: {', '.join(profile['dining_preferences'])}\n消费习惯: {', '.join(profile['spending_habits'])}\n其他方面: {', '.join(profile['other_aspects'])}"
    return profile_str

def preprocess_data(data_list: list[dict], tokenizer: PreTrainedModel) -> tuple[list[str], list[int], list[str], list[str]]:

    def count_tokens(text: str) -> int:
        return tokenizer(text, truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt")['input_ids'].shape[1]
    texts = []
    labels = []
    reasons = []
    users = []
    max_tokens = tokenizer.model_max_length
    for sample in data_list:
        previous_text = ""
        previous_text += "[PROFILE]\n" + format_profile(sample['profile']) + "\n"
        previous_text += "[TASK CONTEXT]\n" + sample['task_context'] + "\n"
        previous_tokens = count_tokens(previous_text)
        history_turns = []
        assistant_turn_idx = 0
        for utt in sample['history']:
            history_turns.append(f"{utt['role']}: {utt['content']}")
            # TODO: truncate it to fit the model input (left the most recent turns)
            current_tokens = previous_tokens + count_tokens("[HISTORY]" + "\n".join(history_turns))
            # Truncate history turns
            while current_tokens > max_tokens and history_turns:
                history_turns.pop(0)
                current_tokens = previous_tokens + count_tokens("[HISTORY]" + "\n".join(history_turns))
            if utt['role'] != 'assistant':
                continue
            final_text = previous_text + "[HISTORY]" + "\n".join(history_turns)
            label = sample['satisfaction_scores'][assistant_turn_idx]
            reason = sample['dissatisfaction_reasons'][assistant_turn_idx]
            texts.append(final_text)
            labels.append(label)
            reasons.append(reason)
            users.append(sample.get("user", "unknown"))
            assistant_turn_idx += 1
    return texts, labels, reasons, users

def evaluate_satisfaction_predictor(model: SatisfactionPredictor, loader: DataLoader) -> dict[str, float]:
    model.eval()
    all_labels = []
    all_pred_scores = []
    all_reasons = []
    all_pred_reason_logits = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", unit="batch"):
            batch = {k: v.to(model.backbone.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels'].float()  # [batch_size]
            reasons = batch['reasons']  # [batch_size]

            ordinal_logits, pred_reason_logits = model(input_ids=input_ids, attention_mask=attention_mask)
            ordinal_probs = torch.sigmoid(ordinal_logits)  # [batch_size, 4]
            ordinal_pred = (ordinal_probs > 0.5).int()
            score = ordinal_pred.sum(dim=1) + 1
            label_score = (labels == 1).sum(dim=1) + 1
            all_labels.extend(label_score.cpu().numpy())
            all_pred_scores.extend(score.cpu().numpy())
            all_reasons.extend(reasons.cpu().numpy())
            all_pred_reason_logits.extend(pred_reason_logits.cpu().numpy())

    # Score metrics
    mae = mean_absolute_error(all_labels, all_pred_scores)
    rmse = root_mean_squared_error(all_labels, all_pred_scores)
    r2 = r2_score(all_labels, all_pred_scores)
    kappa = cohen_kappa_score(all_labels, all_pred_scores, weights='quadratic')
    score_acc = accuracy_score(all_labels, all_pred_scores)
    pearson_corr = pearsonr(all_labels, all_pred_scores)[0]
    spearman_corr = spearmanr(all_labels, all_pred_scores)[0]
    logger.info(f"Score Metrics - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}, Kappa: {kappa:.4f}, Acc: {score_acc:.4f}, Pearson: {pearson_corr:.4f}, Spearman: {spearman_corr:.4f}")
    # Reason metrics
    pred_reason_labels = [logits.argmax() for logits in all_pred_reason_logits]
    reason_acc = accuracy_score(all_reasons, pred_reason_labels)
    f1 = f1_score(all_reasons, pred_reason_labels, average='weighted', zero_division=0)
    logger.info(f"Reason Metrics - Accuracy: {reason_acc:.4f}, F1-weighted: {f1:.4f}")
    return {
        "mae": mae, "rmse": rmse, "r2": r2, "kappa": kappa,
        "score_accuracy": score_acc, "pearson": pearson_corr, "spearman": spearman_corr,
        "reason_accuracy": reason_acc, "reason_f1_weighted": f1,
    }

def monotonic_penalty(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    diff = probs[:, 1:] - probs[:, :-1]
    penalty = torch.relu(diff).mean()
    return penalty

def train_satisfaction_predictor(backbone: PreTrainedModel, train_loader: DataLoader, valid_loader: DataLoader, num_reasons: int, num_epochs: int, alpha: float = 1.0, beta: float = 1.0, gamma: float = 0.1) -> SatisfactionPredictor:
    # train a satisfaction predictor with the training data
    # backbone (e.g., bert-base-chinese) + ordinal head for score regression + classification head for reason classification
    # Loss = α * Regression Loss (e.g., MSE / ordinal loss) + β * Classification Loss (e.g., BCE / cross-entropy)
    model = SatisfactionPredictor(backbone, num_reasons)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    loss_fn_regression = torch.nn.BCEWithLogitsLoss()
    loss_fn_classification = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    eval_results = evaluate_satisfaction_predictor(model, valid_loader)  # Evaluate before training
    torch.save(model.state_dict(), os.path.join('ckpts', 'ordinal', 'best.pt'))  # Save initial model
    best_metric = eval_results["mae"]  # Use MAE as the main metric for model selection
    for epoch in range(num_epochs):
        model.train()
        for batch in (pbar := tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")):
            batch = {k: v.to(model.backbone.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels'].float()  # [batch_size]
            reasons = batch['reasons']  # [batch_size]

            optimizer.zero_grad()
            ordinal_logtis, pred_reason_logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss_regression = loss_fn_regression(ordinal_logtis, labels)
            loss_mono = monotonic_penalty(ordinal_logtis)
            loss_classification = loss_fn_classification(pred_reason_logits, reasons)
            loss = alpha * loss_regression + beta * loss_classification + gamma * loss_mono
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "reg_loss": f"{loss_regression.item():.4f}", "cls_loss": f"{loss_classification.item():.4f}"})
            loss.backward()
            optimizer.step()

        eval_results = evaluate_satisfaction_predictor(model, valid_loader)  # Evaluate after each epoch
        if eval_results["mae"] < best_metric:  # Update best model based on MAE
            best_metric = eval_results["mae"]
            torch.save(model.state_dict(), os.path.join('ckpts', 'ordinal', 'best.pt'))
            logger.info(f"New best model saved with MAE: {best_metric:.4f}")
    # Load the best model before returning
    model.load_state_dict(torch.load(os.path.join('ckpts', 'ordinal', 'best.pt')))
    return model

def main(model_name: str = "bert-base-chinese", batch_size: int = 16, num_epochs: int = 10, eval_checkpoint: str = ""):
    data_list = get_satisfaction_data()
    backbone = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    texts, labels, reasons, users = preprocess_data(data_list, tokenizer)
    reason_to_id = get_reason_to_id()
    num_reasons = len(reason_to_id)
    logger.info(reason_to_id)
    # Split train/valid/test sets (e.g., 80% train, 10% valid, 10% test)
    train_idx, valid_idx, test_idx = split_by_user_group_shuffle_split(users, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    train_reasons = [reasons[i] for i in train_idx]
    valid_texts = [texts[i] for i in valid_idx]
    valid_labels = [labels[i] for i in valid_idx]
    valid_reasons = [reasons[i] for i in valid_idx]
    test_texts = [texts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    test_reasons = [reasons[i] for i in test_idx]
    logger.info(f"Train data: {len(train_texts)}, Valid data: {len(valid_texts)}, Test data: {len(test_texts)}")

    train_dataset = SatisfactionDataset(train_texts, train_labels, train_reasons, tokenizer, reason_to_id)
    valid_dataset = SatisfactionDataset(valid_texts, valid_labels, valid_reasons, tokenizer, reason_to_id)
    test_dataset = SatisfactionDataset(test_texts, test_labels, test_reasons, tokenizer, reason_to_id)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if eval_checkpoint:
        logger.info(f"Loading checkpoint for evaluation: {eval_checkpoint}")
        model = SatisfactionPredictor(backbone, num_reasons)
        model.load_state_dict(torch.load(eval_checkpoint, map_location="cpu"))
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        model = train_satisfaction_predictor(backbone, train_loader, valid_loader, num_reasons, num_epochs, beta=0.5)
    evaluate_satisfaction_predictor(model, test_loader)

def parse_args():
    parser = ArgumentParser(description="Train and evaluate a satisfaction predictor")
    parser.add_argument("-m", "--model_name", type=str, default="bert-base-chinese", help="Pre-trained model name")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("-e", "--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--eval_checkpoint", type=str, default="", help="若指定则跳过训练，直接加载该 checkpoint 在测试集上评测")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
