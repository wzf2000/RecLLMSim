from __future__ import annotations

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from lib.satisfaction_constants import get_dissatisfied_reasons, get_reason_to_id


def build_prompt(row: dict) -> str:
    reason_labels = list(get_reason_to_id().keys())
    reason_labels_text = "、".join(reason_labels)
    dissatisfied_reason_text = "、".join(get_dissatisfied_reasons())
    prompt = (
        "你是一名会进行细粒度对话质量分析的评估员。\n"
        "请基于给定信息先进行推理，再同时预测：\n"
        "1) 当前用户对助手回复的满意度分数（1-5）\n"
        "2) 潜在原因（只有在分数 <=3 时才选择不满意原因；分数 >=4 时必须为“满意”）\n\n"
        f'用户画像：{row["persona"]}\n\n'
        f'任务背景：{row["task_context"]}\n\n'
        f'最近对话历史：{row["history"]}\n\n'
        f'当前助手回复：{row["assistant_reply"]}\n\n'
        f"可选原因标签：{reason_labels_text}\n"
        "原因标签合法性规则：\n"
        f"- 只有当 classification <= 3 时，reason 才能从以下不满意原因中选择：{dissatisfied_reason_text}\n"
        "- 只要 classification >= 4，reason 必须输出“满意”。\n"
        "- 如果 reason 与 classification 不一致，则该输出视为不合法。\n\n"
        "请严格输出一个 JSON 对象，不要输出其他内容：\n"
        '{\n'
        '  "classification": 1-5中的整数,\n'
        '  "reason": "若 classification >= 4 必须输出 满意；若 classification <= 3 只能从其余不满意原因标签中选择一个",\n'
        '  "analysis": "你的详细推理过程"\n'
        "}\n"
    )
    return prompt


def build_messages(prompt: str, model: str) -> list[ChatCompletionMessageParam]:
    if "r1" in model or "reasoner" in model:
        return [
            ChatCompletionUserMessageParam(
                role="user",
                content=prompt,
            )
        ]
    return [
        ChatCompletionSystemMessageParam(
            role="system",
            content="You are a skilled conversational analyst.",
        ),
        ChatCompletionUserMessageParam(
            role="user",
            content=prompt,
        ),
    ]


