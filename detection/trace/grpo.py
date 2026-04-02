"""
在 SFT 模型基础上使用 GRPO（Group Relative Policy Optimization）做 RL post-training。

奖励信号完全来自可验证的 gold 标签（classification 1-5 + reason 标签），无需额外 reward model。

用法：
    accelerate launch detection/grpo_from_sft.py \
      --sft_checkpoint ./detection/ckpts/sft_qwen3_from_gpt5_correct \
      --base_model_name Qwen/Qwen3-8B \
      --output_dir ./detection/ckpts/grpo_from_sft

依赖：pip install trl>=0.18  (需含 GRPOTrainer)
"""

import json
import os
from argparse import ArgumentParser
from typing import Any

import wandb
from accelerate import PartialState
from datasets import Dataset
from loguru import logger
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from .collect_api import get_rows_from_split
from lib.satisfaction_constants import get_reason_to_id
from .sft import (
    build_prompt_like_collect,
    parse_model_json,
    split_history_turns,
)

VALID_REASONS: set[str] = set(get_reason_to_id().keys())


# ---------------------------------------------------------------------------
# 数据集构建：将原始样本转为 GRPOTrainer 所需的 conversational 格式
# ---------------------------------------------------------------------------

def build_grpo_dataset(
    split: str = "train",
    split_seed: int = 42,
    max_history_turns: int = 5,
    limit: int | None = None,
) -> Dataset:
    raw_rows = get_rows_from_split(split=split, split_seed=split_seed)
    if limit is not None:
        raw_rows = raw_rows[:limit]

    records: list[dict[str, Any]] = []
    for row in raw_rows:
        persona = row["persona"]
        task_context = row["task_context"]
        history = row["history"]
        assistant_reply = row["assistant_reply"]

        turns = split_history_turns(history)
        kept = turns[-max_history_turns:] if len(turns) > max_history_turns else turns
        history_text = "\n".join(kept)

        user_prompt = build_prompt_like_collect(persona, task_context, history_text, assistant_reply)
        records.append({
            "prompt": [
                {"role": "system", "content": "You are a skilled conversational analyst."},
                {"role": "user", "content": user_prompt},
            ],
            "gold_score": int(row["gold_score"]),
            "gold_reason": str(row["gold_reason"]),
        })

    ds = Dataset.from_list(records)
    logger.info(f"GRPO dataset built: {len(ds)} samples from split={split}")
    return ds


# ---------------------------------------------------------------------------
# 奖励函数
# ---------------------------------------------------------------------------

def reward_format(completions: list[list[dict]], **kwargs) -> list[float]:
    """JSON 格式合规：能正确解析出 classification (1-5) 和 reason。"""
    rewards: list[float] = []
    for comp in completions:
        text = comp[0]["content"] if comp else ""
        score, reason = parse_model_json(text)
        if score is not None and reason is not None and reason in VALID_REASONS:
            rewards.append(1.0)
        elif score is not None:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def reward_score_accuracy(completions: list[list[dict]], gold_score: list[int], **kwargs) -> list[float]:
    """分数与 gold 的接近程度：精确匹配 1.0，差 1 → 0.5，差 2+ → 0.0。"""
    rewards: list[float] = []
    for comp, gs in zip(completions, gold_score):
        text = comp[0]["content"] if comp else ""
        pred, _ = parse_model_json(text)
        if pred is None:
            rewards.append(0.0)
            continue
        diff = abs(pred - int(gs))
        if diff == 0:
            rewards.append(1.0)
        elif diff == 1:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def reward_reason_accuracy(completions: list[list[dict]], gold_reason: list[str], **kwargs) -> list[float]:
    """原因标签与 gold 完全一致得 1.0。"""
    rewards: list[float] = []
    for comp, gr in zip(completions, gold_reason):
        text = comp[0]["content"] if comp else ""
        _, pred_reason = parse_model_json(text)
        if pred_reason is not None and pred_reason.strip() == str(gr).strip():
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# 模型加载：merge SFT adapter → 作为 GRPO 基座
# ---------------------------------------------------------------------------

def load_merged_sft_model(base_model_name: str, sft_checkpoint: str):
    """加载基座 + SFT LoRA，merge 权重后返回完整模型。"""
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    model = PeftModel.from_pretrained(base, sft_checkpoint)
    model = model.merge_and_unload()
    logger.info(f"Merged SFT adapter from {sft_checkpoint} into {base_model_name}")
    return model


# ---------------------------------------------------------------------------
# 主训练流程
# ---------------------------------------------------------------------------

def train_grpo(
    sft_checkpoint: str,
    base_model_name: str,
    output_dir: str,
    data_split: str = "train",
    split_seed: int = 42,
    max_history_turns: int = 5,
    num_generations: int = 8,
    max_completion_length: int = 1024,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    num_train_epochs: int = 1,
    learning_rate: float = 1e-6,
    lora_r: int = 8,
    lora_alpha: int = 16,
    reward_weights: list[float] | None = None,
    limit: int | None = None,
) -> None:
    base_short = base_model_name.split("/")[-1].lower()  # e.g. "qwen3-8b"
    sft_short = os.path.basename(sft_checkpoint.rstrip("/"))  # e.g. "sft_qwen3_from_gpt5_correct"
    run_name = (
        f"grpo_{base_short}"
        f"_G{num_generations}"
        f"_lr{learning_rate:.0e}"
        f"_r{lora_r}"
        f"_ep{num_train_epochs}"
        f"_{sft_short}"
    )
    if PartialState().is_main_process:
        wandb.init(
            project="satisfaction_prediction_rl",
            name=run_name,
        )

    dataset = build_grpo_dataset(
        split=data_split,
        split_seed=split_seed,
        max_history_turns=max_history_turns,
        limit=limit,
    )

    model = load_merged_sft_model(base_model_name, sft_checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if reward_weights is None:
        reward_weights = [0.2, 0.5, 0.3]

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        report_to="wandb",
        reward_weights=reward_weights,
        log_completions=True,
        # beta=0.001,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=[reward_format, reward_score_accuracy, reward_reason_accuracy],
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"GRPO training finished. Model saved to: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = ArgumentParser(description="GRPO post-training on top of SFT checkpoint")
    p.add_argument("--sft_checkpoint", type=str, required=True, help="SFT 输出目录（含 LoRA adapter）")
    p.add_argument("--base_model_name", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--output_dir", type=str, default="./ckpts/grpo_from_sft")
    p.add_argument("--data_split", type=str, default="train", choices=["train", "valid", "val", "test", "all"])
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--max_history_turns", type=int, default=5)
    p.add_argument("--num_generations", type=int, default=8, help="每个 prompt 采样的 completion 数（G）")
    p.add_argument("--max_completion_length", type=int, default=1024)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-6)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument(
        "--reward_weights",
        type=float,
        nargs=3,
        default=[0.2, 0.5, 0.3],
        metavar=("FMT", "SCORE", "REASON"),
        help="三路 reward 权重：format / score_accuracy / reason_accuracy",
    )
    p.add_argument("--limit", type=int, default=None, help="仅用前 N 条，调试用")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_grpo(
        sft_checkpoint=args.sft_checkpoint,
        base_model_name=args.base_model_name,
        output_dir=args.output_dir,
        data_split=args.data_split,
        split_seed=args.split_seed,
        max_history_turns=args.max_history_turns,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        reward_weights=args.reward_weights,
        limit=args.limit,
    )
