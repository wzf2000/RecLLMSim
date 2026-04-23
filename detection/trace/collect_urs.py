"""
URS 个性化满意度感知 Agent — training-free 推理（session-level）

与 trace/collect_personalized.py 的三阶段流程一致：
  Phase 1: Memory Building（同上；内部自动适配 session-level satisfaction_scores）
  Phase 2: Session Evaluation（改：一次对整段对话打一个 1-5 分）
  Phase 3: Memory Update（保留 per_session / per_session_oracle，不支持 per_turn）

输出 JSONL 字段与 personalized 流水线保持同一 schema（sample_id/user/target_task/
turn_idx/gold_score/pred_score/...），因此 eval/personalized.py、eval/calibrate.py、
eval/diagnose_confusion.py 可直接复用。

sample_id 格式：
  {user}__{target_intent}__{file_stub}__turn_0
  （URS 每个 session 只出 1 条记录；turn_idx 恒为 0，便于与 RecLLMSim 模式兼容）

运行方式（从 detection/ 目录）：
  python trace/collect_urs.py --model gpt-4o --split test \
      --memory_update_mode per_session \
      --output_jsonl outputs/urs/gpt-4o_test_per_session.jsonl
"""

from __future__ import annotations

import json
import os
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Literal

from loguru import logger
from openai import OpenAI
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

from lib.memory import UserMemory, build_memory_update_prompt
from lib.personalized_data import PersonalizedSample, SessionData
from lib.satisfaction_constants import (
    get_reason_to_id,
    is_reason_valid_for_score,
    normalize_reason_for_score,
)
from lib.urs_data import (
    URS_INTENT_LIST,
    build_urs_personalized_samples,
    urs_dataset_stats,
)
from lib.urs_memory import (
    build_session_eval_prompt,
    build_session_eval_prompt_no_memory,
)

# 复用 collect_personalized 的 LLM 调用 / 解析 / 结构化模型
from trace.collect_personalized import (
    StructuredOutputError,
    TurnPrediction,
    _structured_parse,
    build_user_memory,
)
from trace import collect_personalized as _base

MemoryUpdateMode = Literal["none", "per_session", "per_session_oracle"]


# ──────────────────────────────────────────────────────────────────────────────
# 单 session 评估
# ──────────────────────────────────────────────────────────────────────────────

def _normalize_pred_reason(
    pred_score: int,
    pred_reason: str,
    default_reason: str,
    debug_context: str = "",
) -> str:
    normalized = normalize_reason_for_score(
        pred_score,
        pred_reason,
        default_reason=default_reason,
    )
    if not is_reason_valid_for_score(pred_score, pred_reason):
        context = f" for {debug_context}" if debug_context else ""
        logger.warning(
            f"Normalized invalid reason/score pair{context}: "
            f"score={pred_score}, raw_reason={pred_reason} -> {normalized}"
        )
    return normalized


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    before_sleep=before_sleep_log(logger, log_level=40),
)
def _call_predict_session(
    prompt: str,
    model: str,
    debug_context: str = "",
) -> TurnPrediction:
    try:
        return _structured_parse(
            prompt,
            model,
            TurnPrediction,
            temperature=0.3,
            timeout=120,
            system_msg="You are a skilled conversational analyst.",
        )
    except Exception as e:
        dump_dir = "outputs/urs/parse_failures"
        os.makedirs(dump_dir, exist_ok=True)
        safe_context = "".join(
            c if c.isalnum() or c in {"_", "-", "."} else "_"
            for c in (debug_context or "unknown_context")
        )[:160]
        prefix = os.path.join(dump_dir, safe_context)
        meta = {
            "debug_context": debug_context,
            "model": model,
            "prompt_length": len(prompt),
            "error": str(e),
        }
        try:
            with open(prefix + ".json", "w", encoding="utf-8") as fp:
                json.dump(meta, fp, ensure_ascii=False, indent=2)
            with open(prefix + ".prompt.txt", "w", encoding="utf-8") as fp:
                fp.write(prompt)
            if isinstance(e, StructuredOutputError) and e.raw_text:
                with open(prefix + ".raw.txt", "w", encoding="utf-8") as fp:
                    fp.write(e.raw_text)
        except Exception as dump_err:
            logger.warning(f"Failed to dump parse debug info for {debug_context}: {dump_err}")
        raise


def evaluate_urs_session(
    memory: UserMemory | None,
    session: SessionData,
    model: str,
    valid_reasons: set[str],
    default_reason: str,
    block_id: str = "",
) -> list[dict]:
    """对单个 URS session 出 1 条预测。返回形式与 evaluate_session 保持一致（单元素）。"""
    if memory is None:
        prompt = build_session_eval_prompt_no_memory(
            profile=session.profile,
            task_context=session.task_context,
            session_history=session.history,
        )
    else:
        prompt = build_session_eval_prompt(
            memory=memory,
            profile=session.profile,
            task_context=session.task_context,
            session_history=session.history,
        )

    debug_context = (
        f"{block_id}__{os.path.basename(session.file_path)}"
        if block_id else os.path.basename(session.file_path)
    )
    pred = _call_predict_session(prompt, model, debug_context=debug_context)

    pred_reason = _normalize_pred_reason(
        pred.classification,
        pred.reason.strip(),
        default_reason=default_reason,
        debug_context=debug_context,
    )
    if pred_reason not in valid_reasons:
        pred_reason = default_reason

    gold_score = session.satisfaction_scores[0]
    gold_reason = session.dissatisfaction_reasons[0]

    return [{
        "turn_idx": 0,
        "pred_score": pred.classification,
        "pred_reason": pred_reason,
        "gold_score": gold_score,
        "gold_reason": gold_reason,
        "analysis": pred.analysis,
    }]


# ──────────────────────────────────────────────────────────────────────────────
# Memory update（复用 personalized 的 update_memory，但 per_turn 不支持）
# ──────────────────────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    before_sleep=before_sleep_log(logger, log_level=40),
)
def _call_update_memory(prompt: str, model: str):
    from lib.memory import UserMemoryContent
    return _structured_parse(
        prompt, model, UserMemoryContent,
        temperature=0.3, timeout=120,
        system_msg="You are an expert user behavior analyst.",
    )


def update_memory_urs(
    memory: UserMemory,
    session: SessionData,
    turn_predictions: list[dict],
    model: str,
    use_oracle_labels: bool = False,
) -> UserMemory:
    prompt = build_memory_update_prompt(
        existing_memory=memory,
        new_session=session,
        turn_predictions=turn_predictions,
        use_oracle_labels=use_oracle_labels,
    )
    content = _call_update_memory(prompt, model)
    # URS 下 assistant_turns=1，所以 n_history_turns += 1
    return UserMemory.from_content(
        content,
        source_tasks=memory.source_tasks,
        n_history_sessions=memory.n_history_sessions + 1,
        n_history_turns=memory.n_history_turns + 1,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Block-level 推理（一个 PersonalizedSample = 一个 (user, target_intent)）
# ──────────────────────────────────────────────────────────────────────────────

def run_agent_on_urs_sample(
    sample: PersonalizedSample,
    model: str,
    memory_update_mode: MemoryUpdateMode = "per_session",
    save_memory_snapshots: bool = False,
    memory_cache_dir: str | None = None,
    with_memory: bool = True,
) -> list[dict]:
    reason_to_id = get_reason_to_id()
    valid_reasons = set(reason_to_id.keys())
    default_reason = "其它" if "其它" in reason_to_id else next(iter(reason_to_id))

    memory = (
        build_user_memory(sample, model, memory_cache_dir=memory_cache_dir)
        if with_memory else None
    )

    all_records: list[dict] = []

    for session in sample.target_sessions:
        session_file = os.path.basename(session.file_path)
        session_results = evaluate_urs_session(
            memory=memory,
            session=session,
            model=model,
            valid_reasons=valid_reasons,
            default_reason=default_reason,
            block_id=sample.block_id,
        )

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
                "dataset": "urs",
                "chat_model": session.chat_model,
            }
            if memory_snapshot is not None:
                record["memory_snapshot"] = memory_snapshot
            all_records.append(record)

        if with_memory and memory_update_mode in ("per_session", "per_session_oracle"):
            use_oracle = memory_update_mode == "per_session_oracle"
            try:
                memory = update_memory_urs(
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

    return all_records


# ──────────────────────────────────────────────────────────────────────────────
# 主推理流程 + 断点续跑
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


def collect_all_urs(
    samples: list[PersonalizedSample],
    model: str,
    memory_update_mode: MemoryUpdateMode,
    output_jsonl: str,
    max_workers: int,
    save_memory_snapshots: bool,
    memory_cache_dir: str | None,
    with_memory: bool = True,
) -> None:
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    finished_ids = load_finished_ids(output_jsonl)
    logger.info(f"Already finished sample IDs: {len(finished_ids)}")

    output_lock = Lock()

    def process_sample(sample: PersonalizedSample) -> list[dict]:
        expected_ids = {
            f"{sample.user}__{sample.target_task}__{os.path.basename(s.file_path)}__turn_0"
            for s in sample.target_sessions
        }
        if expected_ids and expected_ids.issubset(finished_ids):
            return []

        return run_agent_on_urs_sample(
            sample=sample,
            model=model,
            memory_update_mode=memory_update_mode,
            save_memory_snapshots=save_memory_snapshots,
            memory_cache_dir=memory_cache_dir,
            with_memory=with_memory,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_sample, s): s for s in samples}
        for future in tqdm(as_completed(futures), total=len(futures), desc="urs_blocks"):
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
        description="URS session-level 满意度感知 Agent 推理（training-free）"
    )
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "test", "all"],
    )
    parser.add_argument("--train_ratio", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--memory_update_mode", type=str, default="per_session",
        choices=["none", "per_session", "per_session_oracle"],
        help="per_turn 在 URS 上语义不适用，已禁用",
    )
    parser.add_argument("--output_jsonl", type=str, default="")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--save_memory_snapshots", action="store_true")
    parser.add_argument(
        "--memory_cache_dir", type=str,
        default="outputs/urs/memory_cache",
    )
    parser.add_argument(
        "--target_intents", type=str, nargs="+", default=None,
        choices=URS_INTENT_LIST,
    )
    parser.add_argument(
        "--languages", type=str, nargs="+", default=["zh", "en"],
        choices=["zh", "en"],
        help="数据语言子集（默认 zh+en 一起；可单独跑 zh-only 或 en-only 做消融）",
    )
    parser.add_argument(
        "--min_history_sessions", type=int, default=1,
        help="过滤：历史 session 数量至少为该值（默认 1）",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--limit_users", type=int, default=0)
    parser.add_argument("--user_offset", type=int, default=0)
    parser.add_argument("--no_memory", action="store_true")
    # vLLM 支持
    parser.add_argument("--vllm_base_url", type=str, default="")
    parser.add_argument("--vllm_api_key", type=str, default="EMPTY")
    return parser


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    # vLLM client 切换——改的是 collect_personalized 模块里的 client（_structured_parse 通过它发请求）
    if args.vllm_base_url:
        _base.client = OpenAI(base_url=args.vllm_base_url, api_key=args.vllm_api_key)
        _base._is_vllm = True
        logger.info(f"vLLM mode: base_url={args.vllm_base_url}")

    with_memory = not args.no_memory

    if not args.output_jsonl:
        model_tag = args.model.replace("/", "_").replace(":", "_")
        mode_tag = "no_memory" if not with_memory else args.memory_update_mode
        lang_tag = "_" + "+".join(sorted(args.languages)) if len(args.languages) == 1 else ""
        args.output_jsonl = (
            f"outputs/urs/{model_tag}_{args.split}_{mode_tag}{lang_tag}.jsonl"
        )

    logger.info(f"Model:              {args.model}")
    logger.info(f"Backend:            {'vLLM @ ' + args.vllm_base_url if args.vllm_base_url else 'OpenAI API'}")
    logger.info(f"Split:              {args.split} (train_ratio={args.train_ratio})")
    logger.info(f"Languages:          {args.languages}")
    logger.info(f"With memory:        {with_memory}")
    if with_memory:
        logger.info(f"Memory update mode: {args.memory_update_mode}")
    logger.info(f"Output:             {args.output_jsonl}")
    if with_memory:
        logger.info(f"Memory cache:       {args.memory_cache_dir}")

    samples = build_urs_personalized_samples(
        split=args.split,
        train_ratio=args.train_ratio,
        seed=args.split_seed,
        min_history_sessions=args.min_history_sessions,
        target_intents=args.target_intents,
        languages=tuple(args.languages),
    )

    stats = urs_dataset_stats(samples)
    logger.info(f"Dataset stats: {stats}")

    if args.limit_users > 0:
        users_in_order = list(dict.fromkeys(s.user for s in samples))
        start = max(args.user_offset, 0)
        end = start + args.limit_users
        selected_users = set(users_in_order[start:end])
        samples = [s for s in samples if s.user in selected_users]
        logger.info(f"Limiting to {len(selected_users)} users → {len(samples)} blocks")

    if args.limit > 0:
        samples = samples[: args.limit]
        logger.info(f"Limiting to {len(samples)} blocks for debugging.")

    if with_memory:
        os.makedirs(args.memory_cache_dir, exist_ok=True)

    collect_all_urs(
        samples=samples,
        model=args.model,
        memory_update_mode=args.memory_update_mode,
        output_jsonl=args.output_jsonl,
        max_workers=args.max_workers,
        save_memory_snapshots=args.save_memory_snapshots,
        memory_cache_dir=args.memory_cache_dir if with_memory else None,
        with_memory=with_memory,
    )

    logger.info(f"Done. Results saved to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
