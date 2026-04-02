from transformers import AutoModel, PreTrainedModel

from peft import LoraConfig, get_peft_model


def get_base_model(model_name: str) -> PreTrainedModel:
    return AutoModel.from_pretrained(
        model_name,
        dtype="bfloat16",
        trust_remote_code=True,
    )


def get_model_with_lora(base_model: PreTrainedModel) -> PreTrainedModel:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # Qwen 常用
        lora_dropout=0.1,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    return get_peft_model(base_model, lora_config)
