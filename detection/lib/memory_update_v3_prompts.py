"""User-memory update prompt builders and update evidence helpers."""

from __future__ import annotations

from __future__ import annotations

from .personalized_data import SessionData
from .memory_schema import UserMemoryV3
from .memory_formatting import _truncate


def build_memory_update_prompt_v3(
    existing_memory: UserMemoryV3,
    new_session: SessionData,
    turn_predictions: list[dict],
    use_oracle_labels: bool = False,
) -> str:
    """
    构造 memory update prompt（v3）。

    相比 v2，v3 更强调：
      1. 不要在证据不足时把弱推断改写成硬规则
      2. 优先更新 calibration 与具体反例
      3. 只有新证据明确时才改边界文本
    """
    existing_json = existing_memory.model_dump_json(
        indent=2,
        exclude={"source_tasks", "n_history_sessions", "n_history_turns"},
    )

    session_lines = [
        f"任务：{new_session.task}  背景：{_truncate(new_session.task_context, 150)}",
        "逐轮信息：",
    ]
    assistant_idx = 0
    last_user = ""
    for utt in new_session.history:
        if utt["role"] == "user":
            last_user = _truncate(utt["content"], 120)
        elif utt["role"] == "assistant" and assistant_idx < len(turn_predictions):
            pred = turn_predictions[assistant_idx]
            reply = _truncate(utt["content"])
            pred_s = pred.get("pred_score", "?")
            line = f"  用户：{last_user}\n  助手：{reply}"
            if use_oracle_labels:
                gold_s = pred.get("gold_score", "?")
                gold_r = pred.get("gold_reason", "?")
                line += f"\n  [真实 ★{gold_s}（{gold_r}）]"
            else:
                line += f"\n  [预测 ★{pred_s}]"
            session_lines.append(line)
            assistant_idx += 1

    session_text = "\n".join(session_lines)
    label_note = (
        "本次提供了真实标签，可作为可靠证据更新记忆。"
        if use_oracle_labels
        else "本次仅有模型预测分数（可能有误），请谨慎参考，不要把弱证据升级成硬规则。"
    )

    prompt = (
        "你正在维护一份 v3 用户记忆。请根据新观察到的 session 决定是否需要更新记忆。\n\n"
        f"【现有记忆】\n{existing_json}\n\n"
        f"【新 Session】\n{session_text}\n\n"
        f"【注意】{label_note}\n\n"
        "更新原则：\n"
        "- 优先更新 avg_satisfaction_score / score_distribution 这类 calibration 信息\n"
        "- three_vs_four_distinction / four_vs_five_distinction 只有在新 session 提供了明确相邻分数反例时才修改\n"
        "- 若原本属于“证据不足、只能弱推断”的边界，不要因为单条可疑预测就写成确定性规则\n"
        "- user_specific_requirements 只保留真正改变评分的个性化要求，不要累积通用偏好文本\n"
        "- preferred_response_format 若没有新增信息，可保持极简\n\n"
        "请严格按照原 JSON Schema 输出更新后的记忆（不含程序侧元信息字段），不要输出其他内容。"
    )
    return prompt
