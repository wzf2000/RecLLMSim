"""
SPUR: Supervised Prompting for User satisfaction Rubrics
参考: Lin et al., ACL 2024 (arXiv:2403.12388)

二分类任务：SAT (score >= 4) / DSAT (score <= 3)

三阶段流程：
  Phase 1 — Supervised Extraction：对训练集中每条对话，用 LLM 提取 3 条 rubric 候选
  Phase 2 — Rubric Summarization：将候选 rubric 汇总，让 LLM 归纳为 N 条代表性 rubric
  Phase 3 — Scoring：将 rubric 注入提示词，让 LLM 直接判断测试样本 SAT/DSAT
"""

import json
import os
import random
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    cohen_kappa_score,
)
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

from llm import client
from data_split import split_by_user_group_shuffle_split
from metric_statistics import get_satisfaction_data
from satisfaction_predictor import format_profile

# ──────────────────────────────────────────────────────────────────────────────
# 常量 / 全局设置
# ──────────────────────────────────────────────────────────────────────────────

SAT_LABEL = "SAT"
DSAT_LABEL = "DSAT"


# ──────────────────────────────────────────────────────────────────────────────
# 数据预处理
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# LLM 调用工具
# ──────────────────────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
def _call_llm(messages: list[dict], model: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        timeout=90,
    ).choices[0].message
    content = (response.content or "").strip()
    if content.startswith("<think>"):
        content = content.split("</think>", 1)[1].strip()
    if content.startswith("```json") and content.endswith("```"):
        content = content[7:-3].strip()
    return content


def _sys_user(system: str, user: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1：Supervised Extraction
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2：Rubric Summarization
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3：Scoring / Inference
# ──────────────────────────────────────────────────────────────────────────────

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


def _format_rubrics(rubric_list: list[str]) -> str:
    return "\n".join(f"{i+1}. {r}" for i, r in enumerate(rubric_list))


def _score_one(row: dict, rubrics: dict[str, list[str]], model: str) -> dict:
    """对单条样本进行 rubric 评分，返回预测结果 dict。"""
    sat_rubrics_text = _format_rubrics(rubrics[SAT_LABEL])
    dsat_rubrics_text = _format_rubrics(rubrics[DSAT_LABEL])
    prompt = _SCORING_TMPL.format(
        sat_rubrics=sat_rubrics_text,
        dsat_rubrics=dsat_rubrics_text,
        conversation=format_conversation(row),
    )
    try:
        raw = _call_llm(_sys_user(_SCORING_SYSTEM, prompt), model)
        parsed = json.loads(raw)
        prediction = str(parsed.get("prediction", "")).upper()
        if prediction not in (SAT_LABEL, DSAT_LABEL):
            # fallback：根据 matches 数量决定
            sat_m = len(parsed.get("sat_matches", []))
            dsat_m = len(parsed.get("dsat_matches", []))
            prediction = SAT_LABEL if sat_m >= dsat_m else DSAT_LABEL
        return {
            "gold_score": row["gold_score"],
            "gold_label": row["binary_label"],
            "pred_label": prediction,
            "confidence": float(parsed.get("confidence", 0.5)),
            "sat_matches": parsed.get("sat_matches", []),
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
            "dsat_matches": [],
            "reason": "",
            "parse_ok": False,
        }


def score_test_set(
    rows: list[dict],
    rubrics: dict[str, list[str]],
    model: str,
    cache_file: str = "",
    max_workers: int = 8,
) -> list[dict]:
    """
    Phase 3：并行对测试集每条样本评分，返回结果列表。
    """
    if cache_file and os.path.exists(cache_file):
        logger.info(f"[Phase 3] 加载缓存: {cache_file}")
        results = []
        with open(cache_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        if len(results) == len(rows):
            return results
        logger.info(f"[Phase 3] 缓存不完整 ({len(results)}/{len(rows)})，重新运行")

    results: list[dict | None] = [None] * len(rows)
    lock = Lock()

    def process(i: int, row: dict):
        result = _score_one(row, rubrics, model)
        with lock:
            results[i] = result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process, i, r): i for i, r in enumerate(rows)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Phase 3"):
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
        "dsat_matches": [],
        "reason": "",
        "parse_ok": False,
    } for i, r in enumerate(results)]

    if cache_file:
        os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
        with open(cache_file, "w") as f:
            for r in final:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"[Phase 3] 已保存结果: {cache_file}")

    return final


# ──────────────────────────────────────────────────────────────────────────────
# 评估
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict[str, float]:
    gold = [1 if r["gold_label"] == SAT_LABEL else 0 for r in results]
    pred = [1 if r["pred_label"] == SAT_LABEL else 0 for r in results]
    conf = [r.get("confidence", 0.5) for r in results]
    parse_rate = sum(r.get("parse_ok", False) for r in results) / len(results)

    metrics = {
        "accuracy": accuracy_score(gold, pred),
        "f1_macro": f1_score(gold, pred, average="macro", zero_division=0),
        "f1_sat": f1_score(gold, pred, pos_label=1, average="binary", zero_division=0),
        "f1_dsat": f1_score(gold, pred, pos_label=0, average="binary", zero_division=0),
        "precision_sat": precision_score(gold, pred, pos_label=1, average="binary", zero_division=0),
        "recall_sat": recall_score(gold, pred, pos_label=1, average="binary", zero_division=0),
        "kappa": cohen_kappa_score(gold, pred),
        "auc": roc_auc_score(gold, conf) if len(set(gold)) > 1 else float("nan"),
        "parse_rate": parse_rate,
        "n_samples": len(results),
        "n_sat_gold": int(sum(gold)),
        "n_dsat_gold": int(len(gold) - sum(gold)),
    }
    return metrics


def print_metrics(metrics: dict[str, float], header: str = ""):
    sep = "=" * 55
    if header:
        logger.info(sep)
        logger.info(header)
    logger.info(sep)
    logger.info(f"  样本数:          {metrics['n_samples']}  (SAT={metrics['n_sat_gold']}, DSAT={metrics['n_dsat_gold']})")
    logger.info(f"  解析成功率:      {metrics['parse_rate']:.1%}")
    logger.info(f"  Accuracy:        {metrics['accuracy']:.4f}")
    logger.info(f"  F1-macro:        {metrics['f1_macro']:.4f}")
    logger.info(f"  F1-SAT:          {metrics['f1_sat']:.4f}  (P={metrics['precision_sat']:.4f}, R={metrics['recall_sat']:.4f})")
    logger.info(f"  F1-DSAT:         {metrics['f1_dsat']:.4f}")
    logger.info(f"  Kappa:           {metrics['kappa']:.4f}")
    logger.info(f"  AUC:             {metrics['auc']:.4f}")
    logger.info(sep)


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def main(
    model: str = "gpt-4o",
    k_rubrics: int = 10,
    max_extract_per_label: int = 150,
    max_workers: int = 8,
    output_dir: str = "./outputs/spur",
    limit_test: int = 0,
    seed: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)

    # ── 数据加载与切分 ──────────────────────────────────────────────────────
    logger.info("加载数据...")
    data_list = get_satisfaction_data()
    rows = preprocess_to_rows(data_list)
    logger.info(
        f"总样本: {len(rows)}  SAT={sum(1 for r in rows if r['binary_label']==SAT_LABEL)}"
        f"  DSAT={sum(1 for r in rows if r['binary_label']==DSAT_LABEL)}"
    )

    users = [r["user"] for r in rows]
    train_idx, valid_idx, test_idx = split_by_user_group_shuffle_split(
        users, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=seed
    )
    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]
    logger.info(f"Train={len(train_rows)}, Test={len(test_rows)}")

    if limit_test > 0:
        rng = random.Random(seed)
        test_rows = rng.sample(test_rows, min(limit_test, len(test_rows)))
        logger.info(f"--limit_test 限制测试集为 {len(test_rows)} 条")

    # ── Phase 1: Supervised Extraction ─────────────────────────────────────
    p1_cache = os.path.join(output_dir, "phase1_candidates.json")
    candidates = extract_rubric_candidates(
        train_rows,
        model=model,
        cache_file=p1_cache,
        max_workers=max_workers,
        max_per_label=max_extract_per_label,
    )

    # ── Phase 2: Rubric Summarization ──────────────────────────────────────
    p2_cache = os.path.join(output_dir, f"phase2_rubrics_k{k_rubrics}.json")
    rubrics = summarize_rubrics(
        candidates,
        model=model,
        k=k_rubrics,
        cache_file=p2_cache,
    )

    # ── Phase 3: Scoring ───────────────────────────────────────────────────
    p3_cache = os.path.join(output_dir, f"phase3_results_k{k_rubrics}.jsonl")
    results = score_test_set(
        test_rows,
        rubrics=rubrics,
        model=model,
        cache_file=p3_cache,
        max_workers=max_workers,
    )

    # ── 评估 ────────────────────────────────────────────────────────────────
    metrics = compute_metrics(results)
    print_metrics(metrics, header="SPUR 满意度二分类评估结果")

    metrics_path = os.path.join(output_dir, f"metrics_k{k_rubrics}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"指标已保存至: {metrics_path}")


def parse_args():
    p = ArgumentParser(description="SPUR 用户满意度二分类估计 (Lin et al., ACL 2024)")
    p.add_argument("--model", type=str, default="gpt-4o",
                   help="LLM 模型名称（需与 api_config.json 中的 base_url 兼容）")
    p.add_argument("--k_rubrics", type=int, default=10,
                   help="每个标签归纳的 rubric 数量（论文默认 10）")
    p.add_argument("--max_extract_per_label", type=int, default=150,
                   help="Phase 1 每个标签最多使用的训练样本数（避免 API 调用过多）")
    p.add_argument("--max_workers", type=int, default=8,
                   help="并行 API 调用线程数")
    p.add_argument("--output_dir", type=str, default="./outputs/spur",
                   help="缓存和结果输出目录")
    p.add_argument("--limit_test", type=int, default=0,
                   help="调试用：限制测试集样本数（0=全量）")
    p.add_argument("--seed", type=int, default=42)

    # 跳过某些阶段（用于断点续跑）
    p.add_argument("--skip_phase1", action="store_true",
                   help="跳过 Phase 1，直接从缓存加载候选 rubric（需缓存文件存在）")
    p.add_argument("--skip_phase2", action="store_true",
                   help="跳过 Phase 2，直接从缓存加载汇总 rubric（需缓存文件存在）")
    p.add_argument("--only_eval", action="store_true",
                   help="仅对已有 Phase 3 结果重新计算指标，不调用 LLM")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据（所有模式都需要）
    logger.info("加载数据...")
    data_list = get_satisfaction_data()
    rows = preprocess_to_rows(data_list)
    users = [r["user"] for r in rows]
    train_idx, valid_idx, test_idx = split_by_user_group_shuffle_split(
        users, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=args.seed
    )
    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]
    logger.info(f"Train={len(train_rows)}, Test={len(test_rows)}")

    if args.limit_test > 0:
        rng = random.Random(args.seed)
        test_rows = rng.sample(test_rows, min(args.limit_test, len(test_rows)))
        logger.info(f"测试集限制为 {len(test_rows)} 条")

    p1_cache = os.path.join(args.output_dir, "phase1_candidates.json")
    p2_cache = os.path.join(args.output_dir, f"phase2_rubrics_k{args.k_rubrics}.json")
    p3_cache = os.path.join(args.output_dir, f"phase3_results_k{args.k_rubrics}.jsonl")

    # ── only_eval：直接从 Phase 3 结果计算指标 ──────────────────────────────
    if args.only_eval:
        if not os.path.exists(p3_cache):
            logger.error(f"Phase 3 缓存不存在: {p3_cache}")
            raise SystemExit(1)
        results = []
        with open(p3_cache) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        logger.info(f"从缓存加载 {len(results)} 条结果")
        metrics = compute_metrics(results)
        print_metrics(metrics, header="SPUR 评估结果（仅重新计算指标）")
        raise SystemExit(0)

    # ── Phase 1 ─────────────────────────────────────────────────────────────
    if args.skip_phase1:
        if not os.path.exists(p1_cache):
            logger.error(f"--skip_phase1 指定但缓存不存在: {p1_cache}")
            raise SystemExit(1)
        logger.info(f"[Phase 1] 跳过，从缓存加载: {p1_cache}")
        with open(p1_cache) as f:
            candidates = json.load(f)
    else:
        candidates = extract_rubric_candidates(
            train_rows,
            model=args.model,
            cache_file=p1_cache,
            max_workers=args.max_workers,
            max_per_label=args.max_extract_per_label,
        )

    # ── Phase 2 ─────────────────────────────────────────────────────────────
    if args.skip_phase2:
        if not os.path.exists(p2_cache):
            logger.error(f"--skip_phase2 指定但缓存不存在: {p2_cache}")
            raise SystemExit(1)
        logger.info(f"[Phase 2] 跳过，从缓存加载: {p2_cache}")
        with open(p2_cache) as f:
            rubrics = json.load(f)
    else:
        rubrics = summarize_rubrics(
            candidates,
            model=args.model,
            k=args.k_rubrics,
            cache_file=p2_cache,
        )

    # ── Phase 3 ─────────────────────────────────────────────────────────────
    results = score_test_set(
        test_rows,
        rubrics=rubrics,
        model=args.model,
        cache_file=p3_cache,
        max_workers=args.max_workers,
    )

    # ── 评估 ─────────────────────────────────────────────────────────────────
    metrics = compute_metrics(results)
    print_metrics(metrics, header="SPUR 满意度二分类评估结果")

    metrics_path = os.path.join(args.output_dir, f"metrics_k{args.k_rubrics}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"指标已保存至: {metrics_path}")
