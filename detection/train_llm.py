import pandas as pd
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset
from tenacity import retry, stop_after_attempt, wait_fixed

from prompts import reason_prompts
from reason_data import get_reason_data2

prompt_version = 3

@retry(stop=stop_after_attempt(30), wait=wait_fixed(5))
def get_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-8B",
        max_seq_length=40960,   # Context length - can be longer, but uses more memory
        load_in_4bit=True,     # 4bit uses much less memory
        load_in_8bit=False,      # A bit more accurate, uses 2x memory
        full_finetuning=False,  # We have full finetuning now!
    )
    return model, tokenizer

model, tokenizer = get_model()

model = FastLanguageModel.get_peft_model(
    model,
    r=32,            # Choose any number > 0! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
    lora_alpha=32,   # Best to choose alpha = rank or rank*2
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",     # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,   # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

_, fewshot = get_reason_data2(sample=False, training=False)

dataset = load_dataset("json", data_files={
    "train": "dataset/reasoning_data_v2/train.json",
    "test": "dataset/reasoning_data_v2/test.json",
})

def generate_conversation(examples: dict[str, list]) -> dict[str, list[list[dict[str, str]]]]:
    conversations = []
    prompt = reason_prompts[prompt_version]
    contexts = examples['history']
    profiles = examples['profile']
    ground_truths = examples['ground_truth']
    for context, profile, ground_truth in zip(contexts, profiles, ground_truths):
        if '{profile}' in prompt:
            instruction = prompt.format(context=context, zero_example=fewshot[0], one_example=fewshot[1], two_example=fewshot[2], three_example=fewshot[3], profile=profile)
        else:
            instruction = prompt.format(context=context, zero_example=fewshot[0], one_example=fewshot[1], two_example=fewshot[2], three_example=fewshot[3])
        conversations.append([
            {
                'role': 'user',
                'content': instruction
            },
            {
                'role': 'assistant',
                'content': str(ground_truth)
            }
        ])
    return {
        "conversations": conversations
    }

conversation_dataset = dataset.map(generate_conversation, batched=True)
conversations = tokenizer.apply_chat_template(
    conversation_dataset['train']['conversations'],
    tokenize=False,
)

conversation_series = pd.Series(conversations)
conversation_series.name = "text"
dataset = Dataset.from_pandas(pd.DataFrame(conversation_series))
dataset = dataset.shuffle(seed=3407)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=None,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=1000,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none"
    )
)

trainer_stats = trainer.train()
model.save_pretrained_merged(f"Qwen3-8B-promptV{prompt_version}", tokenizer, save_method="merged_16bit")
