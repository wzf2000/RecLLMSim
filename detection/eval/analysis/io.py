from __future__ import annotations

import json
from pathlib import Path


def load_results(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def round_score(s: float) -> int:
    """预测分数四舍五入到整数（1-5）。"""
    return max(1, min(5, int(round(s))))


def get_task_from_path(file_path: str) -> str:
    """从 file_path 中解析任务名，如 .../User_0/菜谱规划/0.json -> 菜谱规划。"""
    parts = Path(file_path).parts
    if len(parts) >= 2:
        return parts[-2]  # .../TaskName/file.json
    return "unknown"


