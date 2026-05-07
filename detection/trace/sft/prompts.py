from __future__ import annotations

import re
from typing import Any

from transformers import PreTrainedTokenizer

from lib.satisfaction_constants import get_reason_to_id

from .targets import build_assistant_target


def split_history_turns(history: str) -> list[str]:
    """
    按采集端格式划分轮次：每轮以行首的 `user：` / `assistant：` 开头（全角冒号 `：`），
    与 `f'{utt["role"]}：{utt["content"]}\\n'` 一致；单轮 content 内可含任意换行。
    """
    if not (history or "").strip():
        return []
    text = history.strip()
    # 仅在新行行首匹配角色前缀，避免误切单轮内的换行
    pattern = re.compile(r"(?im)^(?P<role>user|assistant)：")
    matches = list(pattern.finditer(text))
    if not matches:
        return [text]
    turns: list[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].rstrip()
        if chunk:
            turns.append(chunk)
    return turns


def build_prompt_like_collect(
    persona: str,
    task_context: str,
    history_text: str,
    assistant_reply: str,
) -> str:
    """与 detection/collect_api_model_traces.py 中 build_prompt 对齐。"""
    reason_labels = list(get_reason_to_id().keys())
    reason_labels_text = "、".join(reason_labels)
    return (
        "你是一名会进行细粒度对话质量分析的评估员。\n"
        "请基于给定信息先进行推理，再同时预测：\n"
        "1) 当前用户对助手回复的满意度分数（1-5）\n"
        "2) 潜在原因（必须从给定标签中选择，包含“满意”）\n\n"
        f"用户画像：{persona}\n\n"
        f"任务背景：{task_context}\n\n"
        f"最近对话历史：{history_text}\n\n"
        f"当前助手回复：{assistant_reply}\n\n"
        f"可选原因标签：{reason_labels_text}\n\n"
        "请严格输出一个 JSON 对象，不要输出其他内容：\n"
        "{\n"
        '  "classification": 1-5中的整数,\n'
        '  "reason": "从可选原因标签中选择一个",\n'
        '  "analysis": "你的详细推理过程"\n'
        "}\n"
    )


def build_source_text(tokenizer: PreTrainedTokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You are a skilled conversational analyst."},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(  # type: ignore
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return f"System: You are a skilled conversational analyst.\nUser: {prompt}\nAssistant:"


def resolve_prompt_for_row(
    row: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    include_reasoning_content: bool,
    max_history_turns_cap: int = 5,
    source_budget_override: int | None = None,
    think_wrap: str = "qwen3",
) -> str:
    """
    优先用 meta 按采集脚本格式重建 prompt；若 source 过长，则减少 history 轮次（保留最近若干轮），
    与采集端「滑动窗口」一致，最多保留 max_history_turns_cap 轮。

    source_budget_override: 推理评测时可传入「prompt 允许的最大 token 数」（例如 max_length - max_new_tokens），
    若设置则不再用教师 target 长度来估算 budget。
    """
    meta = row.get("meta")
    if not isinstance(meta, dict):
        return str(row.get("prompt", ""))

    persona = str(meta.get("persona", "") or "")
    task_context = str(meta.get("task_context", "") or "")
    assistant_reply = str(meta.get("assistant_reply", "") or "")
    turns = split_history_turns(str(meta.get("history", "") or ""))

    if source_budget_override is not None:
        source_budget = max(1, int(source_budget_override))
    else:
        target = build_assistant_target(
            row,
            include_reasoning_content=include_reasoning_content,
            think_wrap=think_wrap,
        )
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
        target_len = len(target_ids)
        if target_len >= max_length:
            # tokenize_example 会只保留 target；prompt 仍给最短合理形式
            k = 0
            history_text = ""
            prompt = build_prompt_like_collect(persona, task_context, history_text, assistant_reply)
            return prompt

        source_budget = max_length - target_len
    # 与采集一致：最多考虑最近 max_history_turns_cap 轮
    max_k = min(max_history_turns_cap, len(turns))
    for k in range(max_k, -1, -1):
        kept_turns = turns[-k:] if k > 0 else []
        history_text = "\n".join(kept_turns)
        prompt = build_prompt_like_collect(persona, task_context, history_text, assistant_reply)
        source_text = build_source_text(tokenizer, prompt)
        src_len = len(tokenizer(source_text, add_special_tokens=False)["input_ids"])
        if src_len <= source_budget:
            return prompt

    # 理论上 k=0 应满足；若仍超出（极长 persona 等），返回最短 prompt，后续仍可在 tokenize 里截断 source
    return build_prompt_like_collect(persona, task_context, "", assistant_reply)
