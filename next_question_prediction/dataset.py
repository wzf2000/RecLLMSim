import os
import pickle
from tqdm import tqdm
from typing import Any
from loguru import logger
from transformers import PreTrainedTokenizer

from utils import conv_format

class NextUserQuestionDataset:
    def __init__(self, data: list[dict[str, Any]], tokenizer: PreTrainedTokenizer, max_length: int, suffix: str, split: str = 'train'):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        if self.tokenizer.chat_template is not None:
            self.conv_format = lambda x: self.tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=False)
        else:
            self.conv_format = lambda x: conv_format(x)
        cached_file_name = f'./cache_data/next_user_question{suffix}_{split}.pkl'
        if os.path.exists(cached_file_name):
            logger.info(f"Loading cached processed data from {cached_file_name}")
            with open(cached_file_name, 'rb') as f:
                self.processed_data = pickle.load(f)
        else:
            logger.info(f"Processing data and caching to {cached_file_name}")
            self._process_data()
            with open(cached_file_name, 'wb') as f:
                pickle.dump(self.processed_data, f)

    def token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True))

    def _process_data(self):
        self.processed_data = []
        for item in tqdm(self.data, desc=f"Processing {self.split} data"):
            history1 = item['origin_history']
            history2 = history1 + [{
                'role': 'user',
                'content': item['user_question']
            }]
            applied_history1 = self.conv_format(history1)
            applied_history2 = self.conv_format(history2)
            while self.token_count(applied_history2) > self.max_length and len(history1) > 1:
                history1 = history1[1:]
                history2 = history2[1:]
                applied_history1 = self.conv_format(history1)
                applied_history2 = self.conv_format(history2)
            if len(history1) <= 1 and self.token_count(applied_history2) > self.max_length:
                continue
            self.processed_data.append((applied_history1, applied_history2))

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        history1, history2 = self.processed_data[idx]
        return {
            'input_ids': history2,
            'labels': history1
        }
