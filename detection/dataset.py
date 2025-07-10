import jieba
import numpy as np
from typing import overload, Literal
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

class ClsDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer: PreTrainedTokenizer, max_length: int = 512):
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

def get_dataset(model_name_or_path: str, train_texts: list[str], train_labels: np.ndarray, val_texts: list[str], val_labels: np.ndarray, test_texts: list[str], test_labels: np.ndarray) -> tuple[ClsDataset, ClsDataset, ClsDataset]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    train_dataset = ClsDataset(train_texts, train_labels, tokenizer)
    val_dataset = ClsDataset(val_texts, val_labels, tokenizer)
    test_dataset = ClsDataset(test_texts, test_labels, tokenizer)
    return train_dataset, val_dataset, test_dataset

def preprocess_data_lm(data_list: list[dict], regression: bool = False, profile: bool = False) -> tuple[list[str], np.ndarray]:
    X = [data['history'] + data['profile'] if profile else data['history'] for data in data_list]
    y = np.array([data['ground_truth'] for data in data_list], dtype=np.float32 if regression else np.int64)
    return X, y

@overload
def preprocess_data_ml(data_list: list[dict], profile: bool = False, labels: Literal[True] = True) -> tuple[list[str], np.ndarray]: ...

@overload
def preprocess_data_ml(data_list: list[dict], profile: bool = False, labels: Literal[False] = False) -> list[str]: ...

def preprocess_data_ml(data_list: list[dict], profile: bool = False, labels: bool = True) -> tuple[list[str], np.ndarray] | list[str]:
    def tokenize(text: str) -> str:
        return ' '.join(jieba.cut(text))

    X = [tokenize(data['history'] + data['profile'] if profile else data['history']) for data in data_list]
    if not labels:
        return X
    y = np.array([data['ground_truth'] for data in data_list])
    return X, y
