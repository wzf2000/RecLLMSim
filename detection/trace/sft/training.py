from __future__ import annotations

import wandb
import torch
from loguru import logger
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from .dataset import get_train_valid_dataset
from .selection import (
    filter_correct_records,
    filter_reflected_wrong_records,
    load_jsonl,
    select_records_for_sft,
)


def train_sft(
    input_jsonl: str,
    model_name: str,
    output_dir: str,
    max_length: int = 2048,
    batch_size: int = 2,
    grad_accum: int = 8,
    num_epochs: int = 10,
    learning_rate: float = 2e-4,
    split_seed: int = 42,
    include_reasoning_content: bool = False,
    max_history_turns: int = 5,
    trace_source: str = "correct_only",
    think_wrap: str = "qwen3",
) -> None:
    wandb.init(
        project="satisfaction_prediction_sft",
        name=(
            f"sft_from_traces_{trace_source}_{think_wrap}_{'with_reasoning' if include_reasoning_content else 'without_reasoning'}"
        ),
    )
    all_rows = load_jsonl(input_jsonl)
    rows = select_records_for_sft(all_rows, trace_source)
    if not rows:
        raise ValueError(
            "No records selected for SFT. "
            "For correct_only use traces with matching prediction/gold; "
            "for correct_plus_reflected_wrong use jsonl from generate_reflections_from_file (wrong rows need reflection.revised_reasoning)."
        )
    n_correct = len(filter_correct_records(all_rows))
    n_reflected = len(filter_reflected_wrong_records(all_rows))
    logger.info(
        f"SFT trace_source={trace_source}, selected={len(rows)} "
        f"(file: {n_correct} correct, {n_reflected} reflected-wrong with revised_reasoning)"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, valid_ds = get_train_valid_dataset(
        rows,
        tokenizer,
        max_length=max_length,
        split_seed=split_seed,
        include_reasoning_content=include_reasoning_content,
        max_history_turns=max_history_turns,
        think_wrap=think_wrap,
    )
    logger.info(f"Train size: {len(train_ds)}, Valid size: {len(valid_ds)}")
    logger.info(f"Include reasoning_content in target: {include_reasoning_content}")
    logger.info(f"think_wrap (assistant target vs Qwen3 infer): {think_wrap}")
    logger.info(f"max_history_turns (cap, align with collect): {max_history_turns}")
    trace_example = train_ds[0]
    decoded_trace_example = tokenizer.decode(trace_example["input_ids"], skip_special_tokens=True)
    logger.info(f"Trace example: {decoded_trace_example}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="wandb",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"SFT finished. Model saved to: {output_dir}")
