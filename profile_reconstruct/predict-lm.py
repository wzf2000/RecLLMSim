import os
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedModel, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from transformers.trainer_utils import EvalPrediction
import numpy as np
from sklearn.model_selection import train_test_split

from data_util import ModelType
from evaluate_util import compute_metrics
from pipe_util import exp_sim, exp_sim2human, exp_sim2human2, exp_human, exp_sim4human, exp_sim4human2, exp_human2sim, exp_human2sim2, exp_sim4human3, exp_sim2human3, exp_sim4human4, exp_sim4human5, HotCold

class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
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
            'labels': torch.FloatTensor(labels)
        }

def get_dataset(model_name_or_path: str, train_texts: list[str], train_labels: np.ndarray, val_texts: list[str], val_labels: np.ndarray, test_texts: list[str], test_labels: np.ndarray) -> tuple[MultiLabelDataset, MultiLabelDataset, MultiLabelDataset]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    train_dataset = MultiLabelDataset(train_texts, train_labels, tokenizer)
    val_dataset = MultiLabelDataset(val_texts, val_labels, tokenizer)
    test_dataset = MultiLabelDataset(test_texts, test_labels, tokenizer)
    return train_dataset, val_dataset, test_dataset

def split_train_val(X: list[str], y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, y_train, X_val, y_val

def compute_metrics_LM(p: EvalPrediction) -> dict[str, float]:
    assert isinstance(p.predictions, np.ndarray)
    assert isinstance(p.label_ids, np.ndarray)
    logits = p.predictions
    labels = p.label_ids

    probs = torch.sigmoid(torch.Tensor(logits)).numpy()
    return compute_metrics(labels, probs)

def get_trainer(model: PreTrainedModel, train_dataset: MultiLabelDataset, val_dataset: MultiLabelDataset):
    training_args = TrainingArguments(
        output_dir=os.path.join(os.path.dirname(__file__), 'results', 'profile'),
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

def work(X_train: list[str], y_train: np.ndarray, X_test: list[str], y_test: np.ndarray, item: str, model_name: str, labels: np.ndarray, **kwargs) -> dict[str, float]:
    X_train, y_train, X_val, y_val = split_train_val(X_train, y_train)
    train_dataset, val_dataset, test_dataset = get_dataset(model_name, X_train, y_train, X_val, y_val, X_test, y_test)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=y_train.shape[1],
        problem_type="multi_label_classification"
    )
    trainer = get_trainer(model, train_dataset, val_dataset)
    trainer.train()
    results = trainer.evaluate(test_dataset)
    return {
        'f1_micro': results['eval_f1_micro'],
        'f1_macro': results['eval_f1_macro'],
        'f1_weighted': results['eval_f1_weighted'],
        'accuracy': results['eval_accuracy'],
        'hit_rate_1': results['eval_hit_rate_1'],
        'hit_rate_3': results['eval_hit_rate_3'],
        'recall_1': results['eval_recall_1'],
        'recall_3': results['eval_recall_3'],
    }

if __name__ == '__main__':
    BERT_BASE = 'bert-base-chinese'
    # exp_sim(BERT_BASE, ModelType.LM, work, 'zh')
    # exp_sim2human(BERT_BASE, ModelType.LM, work)
    # exp_sim4human(BERT_BASE, ModelType.LM, work)
    # exp_human2sim(BERT_BASE, ModelType.LM, work)
    # exp_sim2human2(BERT_BASE, ModelType.LM, work)
    # exp_sim4human2(BERT_BASE, ModelType.LM, work)
    # exp_human2sim2(BERT_BASE, ModelType.LM, work)
    # exp_sim4human3(BERT_BASE, ModelType.LM, work)
    # exp_sim2human3(BERT_BASE, ModelType.LM, work)
    # exp_sim4human4(BERT_BASE, ModelType.LM, work, ratio=0.1)
    # exp_sim4human4(BERT_BASE, ModelType.LM, work, ratio=0.2)
    # exp_sim4human4(BERT_BASE, ModelType.LM, work, ratio=0.5)
    # exp_sim4human4(BERT_BASE, ModelType.LM, work, ratio=1)
    # exp_sim4human4(BERT_BASE, ModelType.LM, work, ratio=2)
    # exp_sim4human4(BERT_BASE, ModelType.LM, work, ratio=5)
    # exp_sim4human4(BERT_BASE, ModelType.LM, work, ratio=10)
    
    # exp_human(BERT_BASE, ModelType.LM, work)
    
    # exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=0.1, hot_cold=HotCold.HOT)
    # exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=0.1, hot_cold=HotCold.COLD)
    # exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=0.1, hot_cold=HotCold.HOT, topk=1)
    # exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=0.1, hot_cold=HotCold.COLD, topk=1)
    # exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=0.1, hot_cold=HotCold.HOT, topk=5)
    # exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=0.1, hot_cold=HotCold.COLD, topk=5)
    # exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=0.5, hot_cold=HotCold.HOT)
    # exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=0.5, hot_cold=HotCold.COLD)
    # exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=0.5, hot_cold=HotCold.HOT, topk=1)
    # exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=0.5, hot_cold=HotCold.COLD, topk=1)
    # exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=0.5, hot_cold=HotCold.HOT, topk=5)
    # exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=0.5, hot_cold=HotCold.COLD, topk=5)
    # exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=1, hot_cold=HotCold.HOT)
    # exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=1, hot_cold=HotCold.COLD)
    # exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=1, hot_cold=HotCold.HOT, topk=1)
    # exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=1, hot_cold=HotCold.COLD, topk=1)
    # exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=1, hot_cold=HotCold.HOT, topk=5)
    # exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=1, hot_cold=HotCold.COLD, topk=5)

    exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=0.1, hot_cold=HotCold.HOT, topk=6)
    exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=0.1, hot_cold=HotCold.COLD, topk=6)
    exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=0.1, hot_cold=HotCold.BOTH, topk=3)
    exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=0.5, hot_cold=HotCold.HOT, topk=6)
    exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=0.5, hot_cold=HotCold.COLD, topk=6)
    exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=0.5, hot_cold=HotCold.BOTH, topk=3)
    exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=1, hot_cold=HotCold.HOT, topk=6)
    exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=1, hot_cold=HotCold.COLD, topk=6)
    exp_sim4human5(BERT_BASE, ModelType.LM, work, ratio=1, hot_cold=HotCold.BOTH, topk=3)
