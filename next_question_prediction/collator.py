import torch
from transformers import PreTrainedTokenizer, BatchEncoding

class UserQuestionCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 8192):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.padding_side = 'left'

    def __call__(self, batch: list[dict]) -> BatchEncoding:
        full_texts = [item['input_ids'] for item in batch]
        input_texts = [item['labels'] for item in batch]

        encodings = self.tokenizer(
            full_texts,
            text_target=input_texts,
            return_tensors='pt',
            padding='longest',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True,
        )
        # truncate to max_length
        if encodings['input_ids'].shape[1] > self.max_length:
            encodings['input_ids'] = encodings['input_ids'][:, -self.max_length:]
            encodings['attention_mask'] = encodings['attention_mask'][:, -self.max_length:]
            encodings['labels'] = encodings['labels'][:, -self.max_length:]
            print("Warning: input truncated to max_length")
        labels = encodings['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[torch.where(encodings["labels"] != self.tokenizer.pad_token_id)] = -100
        encodings["labels"] = labels
        return encodings
