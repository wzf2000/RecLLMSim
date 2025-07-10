import os
from typing import Sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.trainer_utils import EvalPrediction
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data_util import ModelType, get_human_intent_data
from evaluate_util import compute_classification_metrics
from pipe_util import ExpType, split_train_test
from log_util import add_log

class ClsDataset(Dataset):
    def __init__(self, texts: np.ndarray, labels: np.ndarray, tokenizer: PreTrainedTokenizer, max_length: int = 512):
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
            'input_ids': encoding['input_ids'].flatten(), # type: ignore
            'attention_mask': encoding['attention_mask'].flatten(), # type: ignore
            'labels': label
        }

def get_dataset(model_name_or_path: str, train_texts: np.ndarray, train_labels: np.ndarray, val_texts: np.ndarray, val_labels: np.ndarray, test_texts: np.ndarray, test_labels: np.ndarray) -> tuple[ClsDataset, ClsDataset, ClsDataset]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    train_dataset = ClsDataset(train_texts, train_labels, tokenizer)
    val_dataset = ClsDataset(val_texts, val_labels, tokenizer)
    test_dataset = ClsDataset(test_texts, test_labels, tokenizer)
    return train_dataset, val_dataset, test_dataset

def split_train_val(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, y_train, X_val, y_val

def compute_metrics_LM(p: EvalPrediction) -> dict[str, float]:
    assert isinstance(p.predictions, np.ndarray)
    assert isinstance(p.label_ids, np.ndarray)
    logits = p.predictions
    labels = p.label_ids

    predictions = np.argmax(logits, axis=1)
    return compute_classification_metrics(labels, predictions)

def get_trainer(model: PreTrainedModel, train_dataset: ClsDataset, val_dataset: ClsDataset):
    training_args = TrainingArguments(
        output_dir=os.path.join(os.path.dirname(__file__), 'results', 'intent'),
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_LM
    )
    return trainer

def work_human(model_name: str, **kwargs) -> dict[str, float]:
    X, y = get_human_intent_data(model_type=ModelType.ML)
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = split_train_test(X, y) # type: ignore
    X_train, y_train, X_val, y_val = split_train_val(X_train, y_train)
    train_dataset, val_dataset, test_dataset = get_dataset(model_name, X_train, y_train, X_val, y_val, X_test, y_test)
    labels = le.classes_
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label={i: label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
    )
    if 'gpt2' in model_name.lower():
        model.config.pad_token_id = model.config.eos_token_id
    trainer = get_trainer(model, train_dataset, val_dataset)
    trainer.train()
    results = trainer.evaluate(test_dataset)
    metrics = {
        'f1_micro': results['eval_f1_micro'],
        'f1_macro': results['eval_f1_macro'],
        'f1_weighted': results['eval_f1_weighted'],
        'accuracy': results['eval_accuracy'],
    }
    add_log('intent', model_name, ExpType.HUMAN, metrics, cls=True)
    return metrics

if __name__ == '__main__':
    BERT_BASE = 'bert-base-chinese'
    GPT2 = 'IDEA-CCNL/Wenzhong-GPT2-110M'
    # work_human(BERT_BASE)
    work_human(GPT2)
