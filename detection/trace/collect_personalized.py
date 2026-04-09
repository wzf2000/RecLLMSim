"""
个性化满意度感知 Agent — 训练无关推理流水线

三阶段流程：
  Phase 1: Memory Building
    基于历史 session（含满意度标签）构建用户记忆（UserMemory）。
    每个 (用户, 目标任务) block 只需构建一次，可缓存复用。

  Phase 2: Session Evaluation
    对目标 session 中每个 assistant 轮，利用当前记忆预测满意度分数和原因。
    历史窗口大小可配置（默认 5 轮）。

  Phase 3: Memory Update（可选）
    根据 memory_update_mode 参数决定是否及何时更新记忆：
      "none"              — 不更新，记忆在整个 block 内保持不变
      "per_session"       — 每个 target session 预测完成后更新一次（使用模型预测）
      "per_session_oracle"— 每个 target session 预测完成后更新，使用真实标签（oracle 上界）
      "per_turn"          — 每轮预测后立即更新（代价较高）

输出格式（JSONL，每行一个 turn prediction）：
  {
    "sample_id": "User_0__技能学习规划__0.json__turn_0",
    "user": "User_0",
    "target_task": "技能学习规划",
    "target_file": "0.json",
    "turn_idx": 0,               // 在 target session 中的 assistant 轮序号（0-based）
    "gold_score": 4,
    "pred_score": 4,
    "gold_reason": "满意",
    "reason_prediction": "满意",
    "analysis": "...",
    "memory_update_mode": "per_session",
    "model": "gpt-4o",
    "with_memory": true,
    "memory_snapshot": {...}     // 可选，--save_memory_snapshots 时附加
  }

运行方式（从 detection/ 目录）：
  python trace/collect_personalized.py \\
    --model gpt-4o \\
    --split test \\
    --memory_update_mode per_session \\
    --output_jsonl outputs/personalized/test_per_session.jsonl

或通过 scripts/collect_personalized.sh 调用。
"""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

from lib.llm import client
from lib.memory import (
    UserMemory,
    UserMemoryContent,
    build_memory_prompt,
    build_memory_update_prompt,
    build_turn_eval_prompt,
    build_turn_eval_prompt_no_memory,
)
from lib.personalized_data import (
    PersonalizedSample,
    SessionData,
    build_personalized_samples,
    dataset_stats,
)
from lib.satisfaction_constants import get_reason_to_id

MemoryUpdateMode = Literal["none", "per_session", "per_session_oracle", "per_turn"]

# ──────────────────────────────────────────────────────────────────────────────
# LLM 响应模型
# ──────────────────────────────────────────────────────────────────────────────

class TurnPrediction(BaseModel):
    classification: int = Field(ge=1, le=5)
    reason: str
    analysis: str


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Memory Building
# ──────────────────────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    before_sleep=before_sleep_log(logger, log_level=40),
)
def _call_build_memory(prompt: str, model: str) -> UserMemoryContent:
    try:
        response = client.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert user behavior analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,   # memory building 需要稳定输出，降低温度
            response_format=UserMemoryContent,
            timeout=120,
        ).choices[0].message
    except Exception as e:
        logger.error(f"Memory building failed for {model}: {e}")
        raise e
    if response.parsed:
        return response.parsed
    raise RuntimeError(f"Memory building parse failed: {response.refusal or 'unknown error'}")


def build_user_memory(
    sample: PersonalizedSample,
    model: str,
    memory_cache_dir: str | None = None,
) -> UserMemory:
    """
    为给定样本构建用户记忆。

    若 memory_cache_dir 不为 None，则尝试从缓存加载（避免重复调用 LLM）；
    构建成功后也会写入缓存。

    缓存文件名：{user}__{target_task}__{model}.json
    """
    cache_key = f"{sample.user}__{sample.target_task}__{model.replace('/', '_')}"
    cache_path = (
        os.path.join(memory_cache_dir, f"{cache_key}.json")
        if memory_cache_dir
        else None
    )

    # 尝试从缓存加载
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        return UserMemory(**data)

    prompt = build_memory_prompt(
        user_id=sample.user,
        profile=sample.profile,
        history_sessions=sample.history_sessions,
    )
    content = _call_build_memory(prompt, model)
    memory = UserMemory.from_content(
        content,
        source_tasks=sample.history_tasks,
        n_history_sessions=sample.n_history_sessions,
        n_history_turns=sum(s.assistant_turns for s in sample.history_sessions),
    )

    # 写入缓存
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as fp:
            json.dump(memory.model_dump(), fp, ensure_ascii=False, indent=2)

    return memory


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Turn Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    before_sleep=before_sleep_log(logger, log_level=40),
)
def _call_predict_turn(prompt: str, model: str) -> TurnPrediction:
    response = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "You are a skilled conversational analyst."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
        response_format=TurnPrediction,
        timeout=60,
    ).choices[0].message
    if response.parsed:
        return response.parsed
    raise RuntimeError(f"Turn prediction parse failed: {response.refusal or 'unknown error'}")


def evaluate_session(
    memory: UserMemory | None,
    session: SessionData,
    model: str,
    history_window_size: int = 5,
    valid_reasons: set[str] | None = None,
    default_reason: str = "其它",
) -> list[dict]:
    """
    对单个 target session 进行逐轮满意度预测。

    返回每轮的预测结果列表，每条包含：
      pred_score, pred_reason, gold_score, gold_reason, analysis, turn_idx
    """
    if valid_reasons is None:
        reason_to_id = get_reason_to_id()
        valid_reasons = set(reason_to_id.keys())

    results: list[dict] = []
    history_window: list[str] = []
    assistant_turn_idx = 0

    for utt in session.history:
        if utt["role"] == "assistant":
            # 构建 prompt
            if memory is not None:
                prompt = build_turn_eval_prompt(
                    memory=memory,
                    profile=session.profile,
                    task_context=session.task_context,
                    history_window=list(history_window),
                    assistant_reply=utt["content"],
                )
            else:
                prompt = build_turn_eval_prompt_no_memory(
                    profile=session.profile,
                    task_context=session.task_context,
                    history_window=list(history_window),
                    assistant_reply=utt["content"],
                )

            pred = _call_predict_turn(prompt, model)
            pred_reason = pred.reason.strip()
            if pred_reason not in valid_reasons:
                pred_reason = default_reason

            gold_score = session.satisfaction_scores[assistant_turn_idx]
            gold_reason = session.dissatisfaction_reasons[assistant_turn_idx]

            results.append(
                {
                    "turn_idx": assistant_turn_idx,
                    "pred_score": pred.classification,
                    "pred_reason": pred_reason,
                    "gold_score": gold_score,
                    "gold_reason": gold_reason,
                    "analysis": pred.analysis,
                }
            )
            assistant_turn_idx += 1

        # 更新历史窗口（user + assistant 均入窗）
        role_label = "用户" if utt["role"] == "user" else "助手"
        history_window.append(f"{role_label}：{utt['content']}")
        while len(history_window) > history_window_size:
            history_window.pop(0)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Memory Update
# ──────────────────────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    before_sleep=before_sleep_log(logger, log_level=40),
)
def _call_update_memory(prompt: str, model: str) -> UserMemoryContent:
    response = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert user behavior analyst."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        response_format=UserMemoryContent,
        timeout=120,
    ).choices[0].message
    if response.parsed:
        return response.parsed
    raise RuntimeError(f"Memory update parse failed: {response.refusal or 'unknown error'}")


def update_memory(
    memory: UserMemory,
    session: SessionData,
    turn_predictions: list[dict],
    model: str,
    use_oracle_labels: bool = False,
) -> UserMemory:
    """在预测完一个 session 后更新用户记忆。"""
    prompt = build_memory_update_prompt(
        existing_memory=memory,
        new_session=session,
        turn_predictions=turn_predictions,
        use_oracle_labels=use_oracle_labels,
    )
    content = _call_update_memory(prompt, model)
    return UserMemory.from_content(
        content,
        source_tasks=memory.source_tasks,
        n_history_sessions=memory.n_history_sessions + 1,
        n_history_turns=memory.n_history_turns + len(turn_predictions),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Block-level 推理（一个 PersonalizedSample）
# ──────────────────────────────────────────────────────────────────────────────

def run_agent_on_sample(
    sample: PersonalizedSample,
    model: str,
    memory_update_mode: MemoryUpdateMode = "per_session",
    history_window_size: int = 5,
    save_memory_snapshots: bool = False,
    memory_cache_dir: str | None = None,
    with_memory: bool = True,
) -> list[dict]:
    """
    对单个 PersonalizedSample 运行完整 agent 流程，返回所有 turn 的预测结果。

    with_memory=False 时跳过 memory building，使用无记忆 baseline prompt，
    可与 with_memory=True 的结果直接对比（sample_id 相同）。

    每条记录的字段：
      sample_id, user, target_task, target_file, turn_idx,
      gold_score, pred_score, gold_reason, reason_prediction,
      analysis, model, with_memory, memory_update_mode,
      [memory_snapshot]  (可选)
    """
    reason_to_id = get_reason_to_id()
    valid_reasons = set(reason_to_id.keys())
    default_reason = "其它" if "其它" in reason_to_id else next(iter(reason_to_id))

    # Phase 1: Build memory（with_memory=False 时跳过）
    memory = (
        build_user_memory(sample, model, memory_cache_dir=memory_cache_dir)
        if with_memory
        else None
    )

    all_turn_records: list[dict] = []

    for session in sample.target_sessions:
        session_file = os.path.basename(session.file_path)

        if memory_update_mode == "per_turn":
            # 逐轮预测 + 逐轮更新（每轮预测后立即更新记忆）
            session_results = _evaluate_session_per_turn_update(
                memory=memory,
                session=session,
                model=model,
                history_window_size=history_window_size,
                valid_reasons=valid_reasons,
                default_reason=default_reason,
            )
        else:
            # 整个 session 一次性预测
            session_results = evaluate_session(
                memory=memory,
                session=session,
                model=model,
                history_window_size=history_window_size,
                valid_reasons=valid_reasons,
                default_reason=default_reason,
            )

        # 包装为输出记录
        memory_snapshot = memory.model_dump() if (save_memory_snapshots and memory is not None) else None
        for r in session_results:
            record = {
                "sample_id": f"{sample.user}__{sample.target_task}__{session_file}__turn_{r['turn_idx']}",
                "user": sample.user,
                "target_task": sample.target_task,
                "target_file": session_file,
                "turn_idx": r["turn_idx"],
                "gold_score": r["gold_score"],
                "pred_score": r["pred_score"],
                "gold_reason": r["gold_reason"],
                "reason_prediction": r["pred_reason"],
                "analysis": r["analysis"],
                "model": model,
                "with_memory": with_memory,
                "memory_update_mode": memory_update_mode if with_memory else "no_memory",
            }
            if memory_snapshot is not None:
                record["memory_snapshot"] = memory_snapshot
            all_turn_records.append(record)

        # Phase 3: Memory update (仅 with_memory=True 时触发)
        if with_memory and memory_update_mode in ("per_session", "per_session_oracle"):
            use_oracle = memory_update_mode == "per_session_oracle"
            try:
                memory = update_memory(
                    memory=memory,
                    session=session,
                    turn_predictions=session_results,
                    model=model,
                    use_oracle_labels=use_oracle,
                )
            except Exception as e:
                logger.warning(
                    f"Memory update failed for {sample.user}/{session_file}: {e}, "
                    f"keeping existing memory."
                )

    return all_turn_records


def _evaluate_session_per_turn_update(
    memory: UserMemory,
    session: SessionData,
    model: str,
    history_window_size: int,
    valid_reasons: set[str],
    default_reason: str,
) -> list[dict]:
    """
    per_turn 模式：每预测一轮后立即更新记忆。
    由于需要顺序执行，不能并行化。
    """
    results: list[dict] = []
    history_window: list[str] = []
    assistant_turn_idx = 0

    for utt in session.history:
        if utt["role"] == "assistant":
            prompt = build_turn_eval_prompt(
                memory=memory,
                profile=session.profile,
                task_context=session.task_context,
                history_window=list(history_window),
                assistant_reply=utt["content"],
            )
            pred = _call_predict_turn(prompt, model)
            pred_reason = pred.reason.strip()
            if pred_reason not in valid_reasons:
                pred_reason = default_reason

            gold_score = session.satisfaction_scores[assistant_turn_idx]
            gold_reason = session.dissatisfaction_reasons[assistant_turn_idx]

            turn_result = {
                "turn_idx": assistant_turn_idx,
                "pred_score": pred.classification,
                "pred_reason": pred_reason,
                "gold_score": gold_score,
                "gold_reason": gold_reason,
                "analysis": pred.analysis,
            }
            results.append(turn_result)

            # 逐轮更新记忆
            # 构造只含当前轮的 "mini-session"
            mini_session = SessionData(
                user=session.user,
                task=session.task,
                file_path=session.file_path,
                task_context=session.task_context,
                profile=session.profile,
                history=list(history_window) + [utt],  # 包含当前轮
                satisfaction_scores=[gold_score],
                dissatisfaction_reasons=[gold_reason],
                chat_model=session.chat_model,
            )
            try:
                memory = update_memory(
                    memory=memory,
                    session=mini_session,
                    turn_predictions=[turn_result],
                    model=model,
                    use_oracle_labels=False,
                )
            except Exception as e:
                logger.warning(f"Per-turn memory update failed at turn {assistant_turn_idx}: {e}")

            assistant_turn_idx += 1

        role_label = "用户" if utt["role"] == "user" else "助手"
        history_window.append(f"{role_label}：{utt['content']}")
        while len(history_window) > history_window_size:
            history_window.pop(0)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 断点续跑：已完成 sample_id 集合
# ──────────────────────────────────────────────────────────────────────────────

def load_finished_ids(output_jsonl: str) -> set[str]:
    finished: set[str] = set()
    if not os.path.exists(output_jsonl):
        return finished
    with open(output_jsonl, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                finished.add(obj["sample_id"])
            except Exception:
                continue
    return finished


# ──────────────────────────────────────────────────────────────────────────────
# 主推理流程
# ──────────────────────────────────────────────────────────────────────────────

def collect_all(
    samples: list[PersonalizedSample],
    model: str,
    memory_update_mode: MemoryUpdateMode,
    history_window_size: int,
    output_jsonl: str,
    max_workers: int,
    save_memory_snapshots: bool,
    memory_cache_dir: str | None,
    with_memory: bool = True,
) -> None:
    """对所有样本并发执行 agent 推理，结果写入 output_jsonl。"""
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    finished_ids = load_finished_ids(output_jsonl)
    logger.info(f"Already finished turn IDs: {len(finished_ids)}")

    # per_turn 模式需要顺序处理同一 block，设 max_workers 上限不影响正确性
    # 但 block 之间仍可并行
    output_lock = Lock()

    def process_sample(sample: PersonalizedSample) -> list[dict]:
        # 过滤已完成的 block（只要 block 中任一 turn 未完成就重新跑整个 block）
        # 判断依据：block 下所有 turn 的 sample_id 均已存在则跳过
        expected_ids = {
            f"{sample.user}__{sample.target_task}__{os.path.basename(s.file_path)}__turn_{t}"
            for s in sample.target_sessions
            for t in range(s.assistant_turns)
        }
        if expected_ids and expected_ids.issubset(finished_ids):
            return []  # 已全部完成，跳过

        return run_agent_on_sample(
            sample=sample,
            model=model,
            memory_update_mode=memory_update_mode,
            history_window_size=history_window_size,
            save_memory_snapshots=save_memory_snapshots,
            memory_cache_dir=memory_cache_dir,
            with_memory=with_memory,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_sample, s): s for s in samples}
        for future in tqdm(as_completed(futures), total=len(futures), desc="blocks"):
            sample = futures[future]
            try:
                records = future.result()
                if records:
                    with output_lock:
                        with open(output_jsonl, "a", encoding="utf-8") as fp:
                            for r in records:
                                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"Block {sample.block_id} failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> ArgumentParser:
    parser = ArgumentParser(
        description="个性化满意度感知 Agent 推理（training-free）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM 模型名称（默认 gpt-4o）",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "all"],
        help="数据划分（默认 test）",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.2,
        help="训练集用户比例（默认 0.2，即 80%% 用于 test）",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="数据划分随机种子（默认 42）",
    )
    parser.add_argument(
        "--memory_update_mode",
        type=str,
        default="per_session",
        choices=["none", "per_session", "per_session_oracle", "per_turn"],
        help=(
            "记忆更新模式（默认 per_session）：\n"
            "  none              — 不更新，记忆在整个 block 内保持不变\n"
            "  per_session       — 每个 target session 预测后更新（模型预测）\n"
            "  per_session_oracle— 每个 target session 预测后更新（使用真实标签）\n"
            "  per_turn          — 每轮预测后立即更新（最高代价）\n"
        ),
    )
    parser.add_argument(
        "--history_window_size",
        type=int,
        default=5,
        help="评估 prompt 中保留的最近对话轮数（默认 5）",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="",
        help=(
            "输出 JSONL 文件路径；留空时自动生成："
            "outputs/personalized/{model}_{split}_{mode}.jsonl"
        ),
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="并发线程数（默认 8）；per_turn 模式建议降低",
    )
    parser.add_argument(
        "--save_memory_snapshots",
        action="store_true",
        help="在输出记录中附加 memory_snapshot 字段（体积增大，用于分析）",
    )
    parser.add_argument(
        "--memory_cache_dir",
        type=str,
        default="outputs/personalized/memory_cache",
        help="记忆缓存目录（默认 outputs/personalized/memory_cache）",
    )
    parser.add_argument(
        "--target_tasks",
        type=str,
        nargs="+",
        default=None,
        choices=["旅行规划", "礼物准备", "菜谱规划", "技能学习规划"],
        help="限定目标任务类型（默认全部 4 类）",
    )
    parser.add_argument(
        "--min_history_sessions",
        type=int,
        default=1,
        help="过滤：历史 session 数量至少为该值（默认 1）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="最多处理的 block 数量（<=0 表示不限，用于调试）",
    )
    parser.add_argument(
        "--no_memory",
        action="store_true",
        help=(
            "无记忆 baseline 模式：跳过 memory building，"
            "使用与 collect_api.py 相同的无个性化 prompt。"
            "输出 sample_id 与有记忆版本一致，可直接用于 Personalization Gain 计算。"
        ),
    )
    return parser


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    with_memory = not args.no_memory

    # 自动生成输出路径
    if not args.output_jsonl:
        model_tag = args.model.replace("/", "_").replace(":", "_")
        mode_tag = "no_memory" if not with_memory else args.memory_update_mode
        args.output_jsonl = (
            f"outputs/personalized/{model_tag}_{args.split}_{mode_tag}.jsonl"
        )

    logger.info(f"Model:              {args.model}")
    logger.info(f"Split:              {args.split} (train_ratio={args.train_ratio})")
    logger.info(f"With memory:        {with_memory}")
    if with_memory:
        logger.info(f"Memory update mode: {args.memory_update_mode}")
    logger.info(f"History window:     {args.history_window_size} turns")
    logger.info(f"Output:             {args.output_jsonl}")
    if with_memory:
        logger.info(f"Memory cache:       {args.memory_cache_dir}")

    samples = build_personalized_samples(
        split=args.split,
        train_ratio=args.train_ratio,
        seed=args.split_seed,
        min_history_sessions=args.min_history_sessions,
        target_tasks=args.target_tasks,
    )

    stats = dataset_stats(samples)
    logger.info(f"Dataset stats: {stats}")

    if args.limit > 0:
        samples = samples[: args.limit]
        logger.info(f"Limiting to {len(samples)} blocks for debugging.")

    os.makedirs(args.memory_cache_dir, exist_ok=True)

    collect_all(
        samples=samples,
        model=args.model,
        memory_update_mode=args.memory_update_mode,
        history_window_size=args.history_window_size,
        output_jsonl=args.output_jsonl,
        max_workers=args.max_workers,
        save_memory_snapshots=args.save_memory_snapshots,
        memory_cache_dir=args.memory_cache_dir,
        with_memory=with_memory,
    )

    logger.info(f"Done. Results saved to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
