import os
import torch
import random
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model, TaskType
from transformers import PreTrainedTokenizer, PreTrainedModel, Qwen2Tokenizer, Trainer, TrainingArguments
from transformers.models.qwen2 import Qwen2ForCausalLM

from nqp_data import get_nqp_data, get_nqp_data_sim, get_task_list, get_nqp_data_sim_rewritten
from dataset import NextUserQuestionDataset
from collator import UserQuestionCollator

def get_model(model_type: str) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    if model_type == 'Qwen2.5-1.5B':
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(model_name)
        model = Qwen2ForCausalLM.from_pretrained(model_name, dtype='bfloat16')
        model = model.to('cuda')
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model, tokenizer

def get_data(task: str = 'all', augment: float | None = None, truncate_length: int | None = None) -> tuple[list[dict], list[dict]]:
    random.seed(42)
    train_data, test_data = get_nqp_data(sample=False, task_list=get_task_list(human=True, task_name=task))
    if augment is not None and augment > 0:
        sim_data = get_nqp_data_sim_rewritten(task_list=get_task_list(human=False, task_name=task))
        if truncate_length is not None and truncate_length > 0:
            sim_data = [item for item in sim_data if len(item['user_question']) <= truncate_length]
        augment_size = int(len(train_data) * augment)
        if len(sim_data) < augment_size:
            augment_size = len(sim_data)
        train_data.extend(random.sample(sim_data, augment_size))
        random.shuffle(train_data)
        print(f"After augmentation, train data size: {len(train_data)}")
    return train_data, test_data

def test(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, test_data: list[dict], max_length: int = 8192, suffix: str = ''):
    dataset = NextUserQuestionDataset(test_data, tokenizer, max_length, suffix=suffix, split='test')
    collator = UserQuestionCollator(tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator)
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in (pbar := tqdm(dataloader)):
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            labels = batch['labels'].to('cuda')
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            pbar.set_description(f"Loss: {loss.item():.4f}")
            is_nan = torch.isnan(loss)
            if torch.any(is_nan):
                print("NaN loss detected")
    avg_loss = total_loss / len(dataloader)
    print(f"Average Loss: {avg_loss:.6f}")
    if not os.path.exists(f'./ckpt/nqp_model{suffix}'):
        os.makedirs(f'./ckpt/nqp_model{suffix}')
    with open(f'./ckpt/nqp_model{suffix}/test_results.txt', 'w') as f:
        f.write(f'Average Loss: {avg_loss:.6f}\n')

def train(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, train_data: list[dict], max_length: int = 8192, suffix: str = ''):
    dataset = NextUserQuestionDataset(train_data, tokenizer, max_length, suffix=suffix, split='train')
    collator = UserQuestionCollator(tokenizer, max_length)
    training_args = TrainingArguments(
        output_dir=f'./ckpt/nqp_model{suffix}',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        save_steps=500,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=100,
        bf16=True,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=100,
        report_to='none',
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()

def work(model_type: str, max_length: int = 8192, augment: float | None = None, truncate_length: int | None = None, test_only: bool = False, task: str = 'all', **kwargs):
    train_data, test_data = get_data(task=task, augment=augment, truncate_length=truncate_length)
    model, tokenizer = get_model(model_type)
    suffix = f"_{model_type}_maxlen{max_length}"
    if augment is not None:
        suffix += f"_aug{augment}"
    if truncate_length is not None:
        suffix += f"_trunc{truncate_length}"
    if task != 'all':
        suffix += f"_{task}"
    if not test_only:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        model.enable_input_require_grads()
        train(model, tokenizer, train_data, max_length, suffix=suffix)
    test(model, tokenizer, test_data, max_length, suffix=suffix)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='Qwen2.5-1.5B', help='Model type to use')
    parser.add_argument('--max_length', type=int, default=4096, help='Maximum sequence length')
    parser.add_argument('--augment', type=float, default=None, help='Data augmentation ratio')
    parser.add_argument('--test_only', action='store_true', help='Only run testing')
    parser.add_argument('--truncate_length', type=int, default=None, help='Truncate user question length when augmenting data')
    parser.add_argument('--task', type=str, default='all', help='Task to filter, e.g., travel, gift, recipe, skill, all', choices=['travel', 'gift', 'recipe', 'skill', 'all'])
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    work(**vars(args))
