from __future__ import annotations

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from loguru import logger
from tqdm import tqdm

from .constants import DSAT_LABEL, SAT_LABEL
from .data import format_conversation
from .llm import _call_llm, _sys_user


_EXTRACT_SYSTEM = (
    "你是一名对话质量分析专家。"
    "请从给定的对话中提取能够解释用户满意/不满意的关键特征，以简洁、可复用的标准（rubric）形式表达。"
    "每条 rubric 应描述一种通用规律，而非特定对话的细节，字数在 15-40 字以内。"
)

_EXTRACT_SAT_TMPL = """\
以下对话中用户感到【满意】（满意度 >= 4 分）。
请提取 3 条能解释"为何用户满意"的通用规律（rubric），以 JSON 数组形式输出，每条为一个字符串。
不要输出其他内容，只输出 JSON 数组。

{conversation}
"""

_EXTRACT_DSAT_TMPL = """\
以下对话中用户感到【不满意】（满意度 <= 3 分）。
请提取 3 条能解释"为何用户不满意"的通用规律（rubric），以 JSON 数组形式输出，每条为一个字符串。
不要输出其他内容，只输出 JSON 数组。

{conversation}
"""

_SUMMARIZE_SYSTEM = (
    "你是一名对话质量分析专家。"
    "请对以下大量 rubric 候选进行归纳整合，去除重复和过于具体的条目，"
    "提炼出最具代表性、最通用的若干条 rubric。"
    "输出为 JSON 数组，每条为一个字符串，不输出其他内容。"
)

_SUMMARIZE_SAT_TMPL = """\
以下是从多条用户【满意】对话中提取的满意原因候选 rubric，共 {n} 条：

{candidates}

请归纳整合，输出 {k} 条最具代表性的"用户满意"通用 rubric。
每条 rubric 应为通用描述（15-50字），以 JSON 数组输出，不输出其他内容。
"""

_SUMMARIZE_DSAT_TMPL = """\
以下是从多条用户【不满意】对话中提取的不满意原因候选 rubric，共 {n} 条：

{candidates}

请归纳整合，输出 {k} 条最具代表性的"用户不满意"通用 rubric。
每条 rubric 应为通用描述（15-50字），以 JSON 数组输出，不输出其他内容。
"""


def _extract_rubrics_for_one(row: dict, model: str) -> list[str]:
    """对单条样本提取 rubric 候选，返回字符串列表（可能为空）。"""
    tmpl = _EXTRACT_SAT_TMPL if row["binary_label"] == SAT_LABEL else _EXTRACT_DSAT_TMPL
    prompt = tmpl.format(conversation=format_conversation(row))
    try:
        raw = _call_llm(_sys_user(_EXTRACT_SYSTEM, prompt), model)
        candidates = json.loads(raw)
        if isinstance(candidates, list):
            return [str(c).strip() for c in candidates if c]
    except Exception as e:
        logger.warning(f"Rubric extraction failed: {e}")
    return []


def extract_rubric_candidates(
    rows: list[dict],
    model: str,
    cache_file: str = "",
    max_workers: int = 8,
    max_per_label: int = 150,
) -> dict[str, list[str]]:
    """
    Phase 1：并行提取所有训练样本的 rubric 候选。
    返回 {"SAT": [...], "DSAT": [...]}
    """
    if cache_file and os.path.exists(cache_file):
        logger.info(f"[Phase 1] 加载缓存: {cache_file}")
        with open(cache_file) as f:
            return json.load(f)

    sat_rows = [r for r in rows if r["binary_label"] == SAT_LABEL]
    dsat_rows = [r for r in rows if r["binary_label"] == DSAT_LABEL]

    if max_per_label > 0:
        rng = random.Random(42)
        if len(sat_rows) > max_per_label:
            sat_rows = rng.sample(sat_rows, max_per_label)
        if len(dsat_rows) > max_per_label:
            dsat_rows = rng.sample(dsat_rows, max_per_label)

    logger.info(
        f"[Phase 1] Rubric extraction: {len(sat_rows)} SAT + {len(dsat_rows)} DSAT samples"
    )

    candidates: dict[str, list[str]] = {"SAT": [], "DSAT": []}
    lock = Lock()
    all_rows = sat_rows + dsat_rows

    def process(row: dict):
        rubrics = _extract_rubrics_for_one(row, model)
        label = row["binary_label"]
        with lock:
            candidates[label].extend(rubrics)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process, r): i for i, r in enumerate(all_rows)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Phase 1"):
            try:
                fut.result()
            except Exception as e:
                logger.warning(f"Worker failed: {e}")

    logger.info(
        f"[Phase 1] 候选数: SAT={len(candidates['SAT'])}, DSAT={len(candidates['DSAT'])}"
    )

    if cache_file:
        os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(candidates, f, ensure_ascii=False, indent=2)
        logger.info(f"[Phase 1] 已保存缓存: {cache_file}")

    return candidates


def _summarize_one_label(
    candidates: list[str],
    label: str,
    model: str,
    k: int,
    chunk_size: int = 80,
) -> list[str]:
    """
    对单个标签的候选列表做多轮归纳（当候选数超过 chunk_size 时先分块再汇总）。
    """
    tmpl = _SUMMARIZE_SAT_TMPL if label == SAT_LABEL else _SUMMARIZE_DSAT_TMPL

    def _call_summarize(cands: list[str], target_k: int) -> list[str]:
        numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(cands))
        prompt = tmpl.format(n=len(cands), candidates=numbered, k=target_k)
        try:
            raw = _call_llm(_sys_user(_SUMMARIZE_SYSTEM, prompt), model)
            result = json.loads(raw)
            if isinstance(result, list):
                return [str(r).strip() for r in result if r]
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
        # fallback：直接截取候选的前 k 条
        return cands[:target_k]

    if len(candidates) <= chunk_size:
        return _call_summarize(candidates, k)

    # 分块后再次汇总
    logger.info(f"[Phase 2] {label}: {len(candidates)} 条，分块汇总（chunk={chunk_size}）")
    interim: list[str] = []
    for i in range(0, len(candidates), chunk_size):
        chunk = candidates[i : i + chunk_size]
        # 每块保留 k*2 条，最终再归纳到 k
        partial = _call_summarize(chunk, min(k * 2, len(chunk)))
        interim.extend(partial)
        logger.debug(f"  chunk {i//chunk_size}: {len(chunk)} -> {len(partial)}")

    return _call_summarize(interim, k)


def summarize_rubrics(
    candidates: dict[str, list[str]],
    model: str,
    k: int = 10,
    cache_file: str = "",
) -> dict[str, list[str]]:
    """
    Phase 2：将候选 rubric 归纳为各 k 条代表性 rubric。
    返回 {"SAT": [...k条...], "DSAT": [...k条...]}
    """
    if cache_file and os.path.exists(cache_file):
        logger.info(f"[Phase 2] 加载缓存: {cache_file}")
        with open(cache_file) as f:
            return json.load(f)

    rubrics: dict[str, list[str]] = {}
    for label in [SAT_LABEL, DSAT_LABEL]:
        logger.info(f"[Phase 2] 归纳 {label} rubrics（候选={len(candidates[label])}）...")
        rubrics[label] = _summarize_one_label(candidates[label], label, model, k)
        logger.info(f"[Phase 2] {label} rubrics ({len(rubrics[label])}):")
        for i, r in enumerate(rubrics[label], 1):
            logger.info(f"  {i}. {r}")

    if cache_file:
        os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(rubrics, f, ensure_ascii=False, indent=2)
        logger.info(f"[Phase 2] 已保存缓存: {cache_file}")

    return rubrics
