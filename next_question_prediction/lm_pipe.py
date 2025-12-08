import wandb
import numpy as np
from typing import Sequence
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import PreTrainedModel, Trainer, TrainingArguments, AutoTokenizer, PreTrainedTokenizer
from transformers.trainer_utils import EvalPrediction

from evaluate_util import compute_classification_metrics


class ClsDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: np.ndarray, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        input_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
        )
        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length:]
        truncated_text = self.tokenizer.decode(input_ids)

        encoding = self.tokenizer.encode_plus(
            truncated_text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }

def get_dataset(model_name_or_path: str, train_texts: Sequence[str], train_labels: np.ndarray, val_texts: Sequence[str], val_labels: np.ndarray, test_texts: Sequence[str], test_labels: np.ndarray) -> tuple[ClsDataset, ClsDataset, ClsDataset]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    train_dataset = ClsDataset(train_texts, train_labels, tokenizer)
    val_dataset = ClsDataset(val_texts, val_labels, tokenizer)
    test_dataset = ClsDataset(test_texts, test_labels, tokenizer)
    return train_dataset, val_dataset, test_dataset

def split_train_val(X: Sequence[str], y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, y_train, X_val, y_val

def compute_metrics_LM(p: EvalPrediction) -> dict[str, float]:
    assert isinstance(p.predictions, np.ndarray)
    assert isinstance(p.label_ids, np.ndarray)
    logits = p.predictions
    labels = p.label_ids

    predictions = np.argmax(logits, axis=1)
    return compute_classification_metrics(labels, predictions)

def get_trainer(model: PreTrainedModel, train_dataset: ClsDataset, val_dataset: ClsDataset, output_dir: str, learning_rate: float, batch_size: int, epochs: int, run_name: str | None = None) -> Trainer:
    wandb.init(project="next_intent_prediction", name=run_name)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        load_best_model_at_end=True,
        logging_steps=10,
        run_name=run_name,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_LM
    )
    return trainer
