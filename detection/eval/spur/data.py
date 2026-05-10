from __future__ import annotations

from .constants import DSAT_LABEL, SAT_LABEL


def format_profile(profile: dict) -> str:
    return (
        f"性别: {profile.get('gender', '')}\n"
        f"年龄: {profile.get('age', '')}\n"
        f"背景: {profile.get('background', '')}\n"
        f"性格: {', '.join(profile.get('personality', []))}\n"
        f"职业: {profile.get('occupation', '')}\n"
        f"日常兴趣: {', '.join(profile.get('daily_interests', []))}\n"
        f"旅行习惯: {', '.join(profile.get('travel_habits', []))}\n"
        f"饮食偏好: {', '.join(profile.get('dining_preferences', []))}\n"
        f"消费习惯: {', '.join(profile.get('spending_habits', []))}\n"
        f"其他方面: {', '.join(profile.get('other_aspects', []))}"
    )


def preprocess_to_rows(data_list: list[dict]) -> list[dict]:
    """将原始 session 级数据展开为 turn 级行，每条 assistant 发言对应一行。"""
    rows: list[dict] = []
    for sample in data_list:
        persona = format_profile(sample["profile"])
        task_context = sample["task_context"]
        history_window: list[str] = []
        assistant_turn_idx = 0

        for utt in sample["history"]:
            if utt["role"] == "assistant":
                score = int(sample["satisfaction_scores"][assistant_turn_idx])
                rows.append(
                    {
                        "persona": persona,
                        "task_context": task_context,
                        "history": "\n".join(history_window),
                        "assistant_reply": utt["content"],
                        "gold_score": score,
                        "binary_label": SAT_LABEL if score >= 4 else DSAT_LABEL,
                        "user": sample.get("user", "unknown"),
                    }
                )
                assistant_turn_idx += 1

            history_window.append(f'{utt["role"]}：{utt["content"]}\n')
            while len(history_window) > 5:
                history_window.pop(0)

    return rows


def format_conversation(row: dict) -> str:
    """将一行数据格式化为对话文本。"""
    return (
        f"用户画像：{row['persona']}\n\n"
        f"任务背景：{row['task_context']}\n\n"
        f"最近对话历史：{row['history']}\n\n"
        f"当前助手回复：{row['assistant_reply']}"
    )
