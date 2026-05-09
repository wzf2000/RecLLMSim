"""User-memory update prompt builders and update evidence helpers."""

from __future__ import annotations

from __future__ import annotations

from .personalized_data import SessionData
from .memory_schema import UserMemory
from .memory_formatting import _truncate


def build_memory_update_prompt(
    existing_memory: UserMemory,
    new_session: SessionData,
    turn_predictions: list[dict],
    use_oracle_labels: bool = False,
) -> str:
    """
    构造 memory update prompt（v2）。

    策略：保守更新——仅在新 session 提供了与已有记忆明显矛盾或补充的证据时才修改，
    避免预测误差噪声污染已有记忆。
    """
    existing_json = existing_memory.model_dump_json(
        indent=2,
        exclude={"memory_version", "source_tasks", "n_history_sessions", "n_history_turns"},
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
        else "本次仅有模型预测分数（可能有误），请谨慎参考，不要因预测误差大幅修改已有记忆。"
    )

    prompt = (
        "你正在维护一份用户记忆。请根据新观察到的 session 决定是否需要更新记忆。\n\n"
        f"【现有记忆】\n{existing_json}\n\n"
        f"【新 Session】\n{session_text}\n\n"
        f"【注意】{label_note}\n\n"
        "更新原则（保守优先）：\n"
        "- 若新 session 与已有模式一致，保持记忆不变或仅微调\n"
        "- 仅当新 session 提供了明确的反例或补充信息时，才修改 four_vs_five_distinction / "
        "three_vs_four_distinction / user_specific_requirements\n"
        "- 更新 avg_satisfaction_score 和 score_distribution 的统计数字\n"
        "- 可新增 task_specific_observations 条目，但不删除已有条目\n\n"
        "请严格按照原 JSON Schema 输出更新后的记忆（不含元信息字段），不要输出其他内容。"
    )
    return prompt


