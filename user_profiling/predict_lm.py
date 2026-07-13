import os
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedModel, Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizer
from transformers.trainer_utils import EvalPrediction
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize
from argparse import ArgumentParser

from data_util import ModelType
from evaluate_util import compute_metrics
from pipe_util import exp_sim, exp_sim2human, exp_sim2human2, exp_human, exp_sim4human, exp_sim4human2, exp_human2sim, exp_human2sim2, exp_sim4human3, exp_sim2human3, exp_sim4human4, exp_sim4human5, HotCold, set_seed

class MultiLabelDataset(Dataset):
    def __init__(self, texts: list[str], labels: list, tokenizer: PreTrainedTokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
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

def get_dataset(model_name_or_path: str, train_texts: list[str], train_labels: np.ndarray, val_texts: list[str], val_labels: np.ndarray, test_texts: list[str], test_labels: np.ndarray, max_length: int = 512) -> tuple[MultiLabelDataset, MultiLabelDataset, MultiLabelDataset]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    train_dataset = MultiLabelDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = MultiLabelDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = MultiLabelDataset(test_texts, test_labels, tokenizer, max_length)
    return train_dataset, val_dataset, test_dataset

def split_train_val(X: list[str], y: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=seed)
    return X_train, y_train, X_val, y_val

def compute_metrics_LM(p: EvalPrediction) -> dict[str, float]:
    assert isinstance(p.predictions, np.ndarray)
    assert isinstance(p.label_ids, np.ndarray)
    logits = p.predictions
    labels = p.label_ids

    probs = torch.sigmoid(torch.Tensor(logits)).numpy()
    return compute_metrics(labels, probs)

def get_trainer(model: PreTrainedModel, train_dataset: MultiLabelDataset, val_dataset: MultiLabelDataset, ckpt_dir_name: str | None = None, seed: int = 42, epochs: float = 10, batch_size: int = 16) -> Trainer:
    if ckpt_dir_name is not None:
        output_dir = os.path.join(os.path.dirname(__file__), 'results', ckpt_dir_name)
    else:
        output_dir = os.path.join(os.path.dirname(__file__), 'results', 'profile')
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        load_best_model_at_end=True,
        save_total_limit=1,
        seed=seed,
        data_seed=seed,
        report_to='none',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_LM
    )
    return trainer

def work(X_train: list[str], y_train: np.ndarray, X_test: list[str], y_test: np.ndarray, item: str, model_name: str, labels: np.ndarray, ckpt_dir_name: str | None = None, X_val: list[str] | np.ndarray | None = None, y_val: np.ndarray | None = None, seed: int = 42, epochs: float = 10, batch_size: int = 16, max_length: int = 512, **kwargs) -> dict[str, float]:
    if X_val is None or y_val is None:
        X_train, y_train, X_val, y_val = split_train_val(X_train, y_train, seed)
    train_dataset, val_dataset, test_dataset = get_dataset(model_name, X_train, y_train, X_val, y_val, X_test, y_test, max_length)
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=y_train.shape[1],
        problem_type="multi_label_classification"
    )
    trainer = get_trainer(model, train_dataset, val_dataset, ckpt_dir_name, seed, epochs, batch_size)
    trainer.train()
    results = trainer.evaluate(test_dataset)
    return {
        'f1_micro': results['eval_f1_micro'],
        'f1_macro': results['eval_f1_macro'],
        'f1_weighted': results['eval_f1_weighted'],
        'accuracy': results['eval_accuracy'],
        'hit_rate_1': results['eval_hit_rate_1'],
        'hit_rate_3': results['eval_hit_rate_3'],
        'hit_rate_5': results['eval_hit_rate_5'],
        'recall_1': results['eval_recall_1'],
        'recall_3': results['eval_recall_3'],
        'recall_5': results['eval_recall_5'],
        'map_macro': results['eval_map_macro'],
    }

def hc_map(x: str) -> HotCold:
    if x == 'hot':
        return HotCold.HOT
    elif x == 'cold':
        return HotCold.COLD
    elif x == 'both':
        return HotCold.BOTH
    else:
        raise ValueError(f"Invalid value for hot_cold: {x}")

def list_str(x: str) -> list[str]:
    return x.split(',')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-t', '--type', type=str, required=True, choices=['sim', 'sim2human', 'human', 'human2sim', 'sim2human2', 'sim2human3', 'human2sim2', 'sim4human', 'sim4human2', 'sim4human3', 'sim4human4', 'sim4human5'])
    parser.add_argument('-l', '--language', type=str, default='zh', choices=['zh', 'en'])
    parser.add_argument('-d', '--data_version', type=int, default=1, choices=[1, 2, 3, 4], help='1: original data; 2: updated data; 3: updated data for both sim & human; 4: updated data for both sim with rewritten & human')
    parser.add_argument('-c', '--chat_model', type=str, default=None)
    parser.add_argument('-r', '--ratio', type=float, default=1.0, help='Ratio for sim4human4 & sim4human5')
    parser.add_argument('-s', '--samples', type=int, default=-1, help='Number of samples for training human, -1 for all')
    parser.add_argument('--topk', type=int, default=3, help='Top k for sim4human5')
    parser.add_argument('-hc', '--hot_cold', type=hc_map, default=None, choices=[HotCold.HOT, HotCold.COLD, HotCold.BOTH, None], help='Hot or cold for sim4human5')
    parser.add_argument('--only_all', action='store_true', help='Only train & test on all scenarios')
    parser.add_argument('--items', type=list_str, default=None, help='Attributes to train & test on')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=float, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=512)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.type == 'sim':
        exp_sim(args.model, ModelType.LM, work, args.language, only_all=args.only_all, items=args.items)
    elif args.type == 'human':
        exp_human(args.model, ModelType.LM, work, samples=args.samples, data_version=args.data_version, chat_model=args.chat_model, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, max_length=args.max_length, only_all=args.only_all, items=args.items)
    elif args.type == 'sim2human':
        exp_sim2human(args.model, ModelType.LM, work, only_all=args.only_all, items=args.items)
    elif args.type == 'human2sim':
        exp_human2sim(args.model, ModelType.LM, work, only_all=args.only_all, items=args.items)
    elif args.type == 'sim2human2':
        exp_sim2human2(args.model, ModelType.LM, work, only_all=args.only_all, items=args.items)
    elif args.type == 'human2sim2':
        exp_human2sim2(args.model, ModelType.LM, work, only_all=args.only_all, items=args.items)
    elif args.type == 'sim2human3':
        exp_sim2human3(args.model, ModelType.LM, work, data_version=args.data_version, chat_model=args.chat_model, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, max_length=args.max_length, only_all=args.only_all, items=args.items)
    elif args.type == 'sim4human':
        exp_sim4human(args.model, ModelType.LM, work, data_version=args.data_version, chat_model=args.chat_model, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, max_length=args.max_length, only_all=args.only_all, items=args.items)
    elif args.type == 'sim4human2':
        exp_sim4human2(args.model, ModelType.LM, work, data_version=args.data_version, chat_model=args.chat_model, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, max_length=args.max_length, only_all=args.only_all, items=args.items)
    elif args.type == 'sim4human3':
        exp_sim4human3(args.model, ModelType.LM, work, data_version=args.data_version, chat_model=args.chat_model, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, max_length=args.max_length, only_all=args.only_all, items=args.items)
    elif args.type == 'sim4human4':
        exp_sim4human4(args.model, ModelType.LM, work, samples=args.samples, ratio=args.ratio, data_version=args.data_version, chat_model=args.chat_model, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, max_length=args.max_length, only_all=args.only_all, items=args.items)
    elif args.type == 'sim4human5':
        exp_sim4human5(args.model, ModelType.LM, work, samples=args.samples, ratio=args.ratio, hot_cold=args.hot_cold, topk=args.topk, data_version=args.data_version, chat_model=args.chat_model, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, max_length=args.max_length, only_all=args.only_all, items=args.items)
    else:
        raise NotImplementedError
