from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from loguru import logger
from tqdm import tqdm

from .constants import DSAT_LABEL, NEUTRAL_LABEL, SAT_LABEL
from .data import format_conversation
from .llm import _call_llm, _sys_user


_SCORING_SYSTEM = (
    "你是一名对话质量分析专家，擅长根据给定的评分标准判断用户满意度。"
    "请仔细阅读满意/不满意的评判标准（rubric），再对给定对话进行综合判断。"
    "只输出 JSON 对象，不输出其他内容。"
)

_SCORING_TMPL = """\
## 用户满意（SAT）的判断标准（rubric）：
{sat_rubrics}

## 用户不满意（DSAT）的判断标准（rubric）：
{dsat_rubrics}

## 待评估对话：
{conversation}

## 任务：
请根据上述 rubric，对该对话进行逐条核对，并给出最终判断。
输出格式（严格遵守，不输出其他内容）：
{{
  "sat_matches": [符合的SAT rubric编号列表，如 [1,3]],
  "dsat_matches": [符合的DSAT rubric编号列表，如 [2]],
  "prediction": "SAT" 或 "DSAT",
  "confidence": 0.0~1.0,
  "reason": "简短的判断理由（一句话）"
}}
"""

_SCORING_TRINARY_TMPL = """\
## 用户不满意（DSAT, satisfaction score 1--2）的判断标准（rubric）：
{dsat_rubrics}

## 用户一般/中立（NEUTRAL, satisfaction score 3）的判断标准（rubric）：
{neutral_rubrics}

## 用户满意（SAT, satisfaction score 4--5）的判断标准（rubric）：
{sat_rubrics}

## 待评估对话：
{conversation}

## 任务：
请根据上述 rubric，对该对话进行逐条核对，并给出最终三分类判断。
输出格式（严格遵守，不输出其他内容）：
{{
  "dsat_matches": [符合的DSAT rubric编号列表，如 [1,3]],
  "neutral_matches": [符合的NEUTRAL rubric编号列表，如 [2]],
  "sat_matches": [符合的SAT rubric编号列表，如 [1]],
  "prediction": "DSAT" 或 "NEUTRAL" 或 "SAT",
  "confidence": 0.0~1.0,
  "reason": "简短的判断理由（一句话）"
}}
"""


def _format_rubrics(rubric_list: list[str]) -> str:
    return "\n".join(f"{i+1}. {r}" for i, r in enumerate(rubric_list))


def _fallback_trinary_prediction(parsed: dict) -> str:
    counts = {
        DSAT_LABEL: len(parsed.get("dsat_matches", [])),
        NEUTRAL_LABEL: len(parsed.get("neutral_matches", [])),
        SAT_LABEL: len(parsed.get("sat_matches", [])),
    }
    best = max(counts.values())
    winners = [label for label, count in counts.items() if count == best]
    if NEUTRAL_LABEL in winners:
        return NEUTRAL_LABEL
    return SAT_LABEL if SAT_LABEL in winners else DSAT_LABEL


def _score_one(
    row: dict,
    rubrics: dict[str, list[str]],
    model: str,
    label_schema: str = "binary",
) -> dict:
    """对单条样本进行 rubric 评分，返回预测结果 dict。"""
    if label_schema == "trinary":
        prompt = _SCORING_TRINARY_TMPL.format(
            dsat_rubrics=_format_rubrics(rubrics[DSAT_LABEL]),
            neutral_rubrics=_format_rubrics(rubrics[NEUTRAL_LABEL]),
            sat_rubrics=_format_rubrics(rubrics[SAT_LABEL]),
            conversation=format_conversation(row),
        )
        valid_labels = (DSAT_LABEL, NEUTRAL_LABEL, SAT_LABEL)
    else:
        prompt = _SCORING_TMPL.format(
            sat_rubrics=_format_rubrics(rubrics[SAT_LABEL]),
            dsat_rubrics=_format_rubrics(rubrics[DSAT_LABEL]),
            conversation=format_conversation(row),
        )
        valid_labels = (SAT_LABEL, DSAT_LABEL)
    try:
        raw = _call_llm(_sys_user(_SCORING_SYSTEM, prompt), model)
        parsed = json.loads(raw)
        prediction = str(parsed.get("prediction", "")).upper()
        if prediction not in valid_labels:
            # fallback：根据 matches 数量决定
            if label_schema == "trinary":
                prediction = _fallback_trinary_prediction(parsed)
            else:
                sat_m = len(parsed.get("sat_matches", []))
                dsat_m = len(parsed.get("dsat_matches", []))
                prediction = SAT_LABEL if sat_m >= dsat_m else DSAT_LABEL
        return {
            "gold_score": row["gold_score"],
            "gold_label": row["binary_label"],
            "pred_label": prediction,
            "confidence": float(parsed.get("confidence", 0.5)),
            "sat_matches": parsed.get("sat_matches", []),
            "neutral_matches": parsed.get("neutral_matches", []),
            "dsat_matches": parsed.get("dsat_matches", []),
            "reason": parsed.get("reason", ""),
            "parse_ok": True,
        }
    except Exception as e:
        logger.warning(f"Scoring failed: {e}")
        return {
            "gold_score": row["gold_score"],
            "gold_label": row["binary_label"],
            "pred_label": DSAT_LABEL,
            "confidence": 0.5,
            "sat_matches": [],
            "neutral_matches": [],
            "dsat_matches": [],
            "reason": "",
            "parse_ok": False,
        }


def score_rows(
    rows: list[dict],
    rubrics: dict[str, list[str]],
    model: str,
    cache_file: str = "",
    max_workers: int = 8,
    desc: str = "Phase 3",
    label_schema: str = "binary",
) -> list[dict]:
    """对任意一批样本并行进行 rubric 评分，返回结果列表。"""
    if cache_file and os.path.exists(cache_file):
        logger.info(f"[{desc}] 加载缓存: {cache_file}")
        cached = []
        with open(cache_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    cached.append(json.loads(line))
        if len(cached) == len(rows):
            return cached
        logger.info(f"[{desc}] 缓存不完整 ({len(cached)}/{len(rows)})，重新运行")

    results: list[dict | None] = [None] * len(rows)
    lock = Lock()

    def process(i: int, row: dict):
        result = _score_one(row, rubrics, model, label_schema=label_schema)
        with lock:
            results[i] = result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process, i, r): i for i, r in enumerate(rows)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            try:
                fut.result()
            except Exception as e:
                logger.warning(f"Worker failed: {e}")

    final = [r if r is not None else {
        "gold_score": rows[i]["gold_score"],
        "gold_label": rows[i]["binary_label"],
        "pred_label": DSAT_LABEL,
        "confidence": 0.5,
        "sat_matches": [],
        "neutral_matches": [],
        "dsat_matches": [],
        "reason": "",
        "parse_ok": False,
    } for i, r in enumerate(results)]

    if cache_file:
        os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
        with open(cache_file, "w") as f:
            for r in final:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"[{desc}] 已保存结果: {cache_file}")

    return final


def score_test_set(
    rows: list[dict],
    rubrics: dict[str, list[str]],
    model: str,
    cache_file: str = "",
    max_workers: int = 8,
    label_schema: str = "binary",
) -> list[dict]:
    """Phase 3 测试集评分（向后兼容的 wrapper）。"""
    return score_rows(rows, rubrics, model, cache_file, max_workers, desc="Phase 3 (test)", label_schema=label_schema)
