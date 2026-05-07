from __future__ import annotations

from predictor.bert import format_profile

from .constants import DSAT_LABEL, SAT_LABEL


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
