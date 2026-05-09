from __future__ import annotations

import json
import os

from loguru import logger


def load_records(path: str) -> list[dict]:
    """从 JSONL 加载预测结果，仅保留含 gold_score 和 pred_score 的记录。"""
    records: list[dict] = []
    if not os.path.exists(path):
        logger.warning(f"文件不存在: {path}")
        return records
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("gold_score") is not None and r.get("pred_score") is not None:
                    records.append(r)
            except json.JSONDecodeError:
                continue
    return records


# ──────────────────────────────────────────────────────────────────────────────
# 全局指标
# ──────────────────────────────────────────────────────────────────────────────

