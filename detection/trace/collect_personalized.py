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

import os
import sys
from argparse import ArgumentParser
from typing import Literal

from loguru import logger
from pydantic import BaseModel

from openai import OpenAI

_DETECTION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DETECTION_DIR not in sys.path:
    sys.path.insert(0, _DETECTION_DIR)

from lib.anchor_retrieval import AnchorRetriever, AnchorTurn
from lib.llm import client as _default_client
from lib.memory import (
    UserMemory,
    UserMemoryV3,
)
from lib.personalized_data import (
    PersonalizedSample,
    SessionData,
    build_personalized_samples,
    dataset_stats,
)
from trace.personalized_memory import (
    build_user_memory as _build_user_memory_impl,
    update_memory as _update_memory_impl,
)
from trace.personalized_predictions import (
    BoundaryTurnPrediction,
    DsatRefinementPrediction,
    EpisodicBoundaryRefinementPrediction,
    HistoryPriorDeltaPrediction,
    HistoryPriorDeltaV2Prediction,
    SatRefinementPrediction,
    SelectiveBoundaryTurnPrediction,
    TurnPrediction,
)
from trace.structured_output import (
    StructuredOutputError,
    structured_parse,
    structured_parse_from_raw_text,
)
from trace.personalized_turn_eval import (
    call_predict_turn as _call_predict_turn_impl,
    evaluate_session as _evaluate_session_impl,
    predict_turn_fullscale_from_boundary_v2 as _predict_turn_fullscale_from_boundary_v2_impl,
    predict_turn_v3_two_stage as _predict_turn_v3_two_stage_impl,
    predict_turn_v3_two_stage_v2 as _predict_turn_v3_two_stage_v2_impl,
    predict_turn_with_optional_selective_refute as _predict_turn_with_optional_selective_refute_impl,
)
from trace.personalized_collect import (
    collect_all as _collect_all_impl,
    load_finished_ids as _load_finished_ids_impl,
)
from trace.personalized_runner import (
    evaluate_session_per_turn_update as _evaluate_session_per_turn_update_impl,
    run_agent_on_sample as _run_agent_on_sample_impl,
)

MemoryUpdateMode = Literal["none", "per_session", "per_session_oracle", "per_turn"]
MemoryVersion = Literal["v2", "v3"]
MemoryUpdatePromptVersion = Literal["auto", "v2", "v2_1", "v2_2", "v2_3", "v2_4", "v2_5", "v3"]

# ──────────────────────────────────────────────────────────────────────────────
# LLM 客户端（可在 main() 中切换为 vLLM client）
# ──────────────────────────────────────────────────────────────────────────────

client = _default_client   # module-level，可被 main() 替换为 vLLM client
memory_client = _default_client
_is_vllm: bool = False     # 仅用于日志标识

def _structured_parse(
    prompt: str,
    model: str,
    response_model: type[BaseModel],
    temperature: float = 0.3,
    timeout: int = 120,
    system_msg: str = "You are an expert user behavior analyst.",
) -> BaseModel:
    return structured_parse(
        client=client,
        prompt=prompt,
        model=model,
        response_model=response_model,
        temperature=temperature,
        timeout=timeout,
        system_msg=system_msg,
    )


def _structured_parse_from_raw_text(
    prompt: str,
    model: str,
    response_model: type[BaseModel],
    temperature: float = 0.3,
    timeout: int = 120,
    system_msg: str = "You are an expert user behavior analyst.",
) -> BaseModel:
    return structured_parse_from_raw_text(
        client=client,
        prompt=prompt,
        model=model,
        response_model=response_model,
        temperature=temperature,
        timeout=timeout,
        system_msg=system_msg,
    )


def _structured_parse_memory(
    prompt: str,
    model: str,
    response_model: type[BaseModel],
    temperature: float = 0.3,
    timeout: int = 120,
    system_msg: str = "You are an expert user behavior analyst.",
) -> BaseModel:
    return structured_parse(
        client=memory_client,
        prompt=prompt,
        model=model,
        response_model=response_model,
        temperature=temperature,
        timeout=timeout,
        system_msg=system_msg,
    )


def build_user_memory(
    sample: PersonalizedSample,
    model: str,
    memory_cache_dir: str | None = None,
    memory_version: MemoryVersion = "v2",
) -> UserMemory | UserMemoryV3:
    return _build_user_memory_impl(
        sample=sample,
        model=model,
        parse_fn=_structured_parse_memory,
        memory_cache_dir=memory_cache_dir,
        memory_version=memory_version,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Turn Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def _call_predict_turn(
    prompt: str,
    model: str,
    prompt_version: str = "v2",
    debug_context: str = "",
) -> (
    TurnPrediction
    | BoundaryTurnPrediction
    | SelectiveBoundaryTurnPrediction
    | HistoryPriorDeltaPrediction
    | HistoryPriorDeltaV2Prediction
    | SatRefinementPrediction
    | DsatRefinementPrediction
    | EpisodicBoundaryRefinementPrediction
):
    return _call_predict_turn_impl(
        prompt=prompt,
        model=model,
        parse_fn=_structured_parse,
        raw_parse_fn=_structured_parse_from_raw_text,
        prompt_version=prompt_version,
        debug_context=debug_context,
    )


def _predict_turn_with_optional_selective_refute(
    memory: UserMemory | None,
    session: SessionData,
    model: str,
    history_window: list[str],
    assistant_reply: str,
    turn_eval_prompt_version: str,
    debug_context: str,
    default_reason: str,
    anchors: list[AnchorTurn] | None = None,
) -> dict:
    return _predict_turn_with_optional_selective_refute_impl(
        memory=memory,
        session=session,
        model=model,
        history_window=history_window,
        assistant_reply=assistant_reply,
        turn_eval_prompt_version=turn_eval_prompt_version,
        debug_context=debug_context,
        default_reason=default_reason,
        call_predict_fn=_call_predict_turn,
        anchors=anchors,
    )


def _predict_turn_fullscale_from_boundary_v2(
    memory: UserMemory | None,
    session: SessionData,
    model: str,
    history_window: list[str],
    assistant_reply: str,
    debug_context: str,
    default_reason: str,
    anchors: list[AnchorTurn] | None = None,
) -> dict:
    return _predict_turn_fullscale_from_boundary_v2_impl(
        memory=memory,
        session=session,
        model=model,
        history_window=history_window,
        assistant_reply=assistant_reply,
        debug_context=debug_context,
        default_reason=default_reason,
        call_predict_fn=_call_predict_turn,
        anchors=anchors,
    )


def _predict_turn_v3_two_stage(
    memory: UserMemory | UserMemoryV3 | None,
    session: SessionData,
    model: str,
    history_window: list[str],
    assistant_reply: str,
    debug_context: str,
    default_reason: str,
    anchors: list[AnchorTurn] | None = None,
) -> dict:
    return _predict_turn_v3_two_stage_impl(
        memory=memory,
        session=session,
        model=model,
        history_window=history_window,
        assistant_reply=assistant_reply,
        debug_context=debug_context,
        default_reason=default_reason,
        call_predict_fn=_call_predict_turn,
        anchors=anchors,
    )


def _predict_turn_v3_two_stage_v2(
    memory: UserMemory | UserMemoryV3 | None,
    session: SessionData,
    model: str,
    history_window: list[str],
    assistant_reply: str,
    debug_context: str,
    default_reason: str,
    anchors: list[AnchorTurn] | None = None,
) -> dict:
    return _predict_turn_v3_two_stage_v2_impl(
        memory=memory,
        session=session,
        model=model,
        history_window=history_window,
        assistant_reply=assistant_reply,
        debug_context=debug_context,
        default_reason=default_reason,
        call_predict_fn=_call_predict_turn,
        anchors=anchors,
    )


def evaluate_session(
    memory: UserMemory | None,
    session: SessionData,
    model: str,
    history_window_size: int = 5,
    valid_reasons: set[str] | None = None,
    default_reason: str = "其它",
    retriever: AnchorRetriever | None = None,
    n_anchors: int = 0,
    turn_eval_prompt_version: str = "v2",
    block_id: str = "",
) -> list[dict]:
    return _evaluate_session_impl(
        memory=memory,
        session=session,
        model=model,
        call_predict_fn=_call_predict_turn,
        history_window_size=history_window_size,
        valid_reasons=valid_reasons,
        default_reason=default_reason,
        retriever=retriever,
        n_anchors=n_anchors,
        turn_eval_prompt_version=turn_eval_prompt_version,
        block_id=block_id,
    )


def update_memory(
    memory: UserMemory | UserMemoryV3,
    session: SessionData,
    turn_predictions: list[dict],
    model: str,
    use_oracle_labels: bool = False,
    memory_version: MemoryVersion = "v2",
    memory_update_prompt_version: MemoryUpdatePromptVersion = "auto",
) -> UserMemory | UserMemoryV3:
    return _update_memory_impl(
        memory=memory,
        session=session,
        turn_predictions=turn_predictions,
        model=model,
        parse_fn=_structured_parse_memory,
        use_oracle_labels=use_oracle_labels,
        memory_version=memory_version,
        memory_update_prompt_version=memory_update_prompt_version,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Block-level 推理（一个 PersonalizedSample）
# ──────────────────────────────────────────────────────────────────────────────

def run_agent_on_sample(
    sample: PersonalizedSample,
    model: str,
    memory_update_mode: MemoryUpdateMode = "per_session",
    memory_version: MemoryVersion = "v2",
    memory_update_prompt_version: MemoryUpdatePromptVersion = "auto",
    history_window_size: int = 5,
    save_memory_snapshots: bool = False,
    memory_cache_dir: str | None = None,
    with_memory: bool = True,
    n_anchors: int = 0,
    turn_eval_prompt_version: str = "v2",
    memory_model: str | None = None,
) -> list[dict]:
    return _run_agent_on_sample_impl(
        sample=sample,
        model=model,
        build_user_memory_fn=build_user_memory,
        evaluate_session_fn=evaluate_session,
        update_memory_fn=update_memory,
        predict_turn_with_optional_selective_refute_fn=_predict_turn_with_optional_selective_refute,
        predict_turn_fullscale_from_boundary_v2_fn=_predict_turn_fullscale_from_boundary_v2,
        predict_turn_v3_two_stage_fn=_predict_turn_v3_two_stage,
        predict_turn_v3_two_stage_v2_fn=_predict_turn_v3_two_stage_v2,
        memory_update_mode=memory_update_mode,
        memory_version=memory_version,
        memory_update_prompt_version=memory_update_prompt_version,
        history_window_size=history_window_size,
        save_memory_snapshots=save_memory_snapshots,
        memory_cache_dir=memory_cache_dir,
        with_memory=with_memory,
        n_anchors=n_anchors,
        turn_eval_prompt_version=turn_eval_prompt_version,
        memory_model=memory_model,
    )


def _evaluate_session_per_turn_update(
    memory: UserMemory | UserMemoryV3,
    session: SessionData,
    model: str,
    history_window_size: int,
    valid_reasons: set[str],
    default_reason: str,
    retriever: AnchorRetriever | None = None,
    n_anchors: int = 0,
    turn_eval_prompt_version: str = "v2",
    block_id: str = "",
    memory_version: MemoryVersion = "v2",
    memory_update_prompt_version: MemoryUpdatePromptVersion = "auto",
    memory_model: str | None = None,
) -> list[dict]:
    return _evaluate_session_per_turn_update_impl(
        memory=memory,
        session=session,
        model=model,
        memory_model=memory_model or model,
        history_window_size=history_window_size,
        valid_reasons=valid_reasons,
        default_reason=default_reason,
        update_memory_fn=update_memory,
        predict_turn_with_optional_selective_refute_fn=_predict_turn_with_optional_selective_refute,
        predict_turn_fullscale_from_boundary_v2_fn=_predict_turn_fullscale_from_boundary_v2,
        predict_turn_v3_two_stage_fn=_predict_turn_v3_two_stage,
        predict_turn_v3_two_stage_v2_fn=_predict_turn_v3_two_stage_v2,
        retriever=retriever,
        n_anchors=n_anchors,
        turn_eval_prompt_version=turn_eval_prompt_version,
        block_id=block_id,
        memory_version=memory_version,
        memory_update_prompt_version=memory_update_prompt_version,
    )


def load_finished_ids(output_jsonl: str) -> set[str]:
    return _load_finished_ids_impl(output_jsonl)


def collect_all(
    samples: list[PersonalizedSample],
    model: str,
    memory_update_mode: MemoryUpdateMode,
    memory_version: MemoryVersion,
    memory_update_prompt_version: MemoryUpdatePromptVersion,
    history_window_size: int,
    output_jsonl: str,
    max_workers: int,
    save_memory_snapshots: bool,
    memory_cache_dir: str | None,
    with_memory: bool = True,
    n_anchors: int = 0,
    turn_eval_prompt_version: str = "v2",
    memory_model: str | None = None,
) -> None:
    return _collect_all_impl(
        samples=samples,
        model=model,
        memory_update_mode=memory_update_mode,
        memory_version=memory_version,
        memory_update_prompt_version=memory_update_prompt_version,
        history_window_size=history_window_size,
        output_jsonl=output_jsonl,
        max_workers=max_workers,
        save_memory_snapshots=save_memory_snapshots,
        memory_cache_dir=memory_cache_dir,
        run_agent_on_sample_fn=run_agent_on_sample,
        with_memory=with_memory,
        n_anchors=n_anchors,
        turn_eval_prompt_version=turn_eval_prompt_version,
        memory_model=memory_model,
    )


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
        "--memory_model",
        type=str,
        default="",
        help=(
            "用于构建/更新用户记忆的模型。留空时与 --model 相同；"
            "设置后 --model 仅用于 turn-level 预测。"
        ),
    )
    parser.add_argument(
        "--memory_version",
        type=str,
        default="v2",
        choices=["v2", "v3"],
        help="用户记忆版本（默认 v2；v3 会启用更保守的证据充分性建模）",
    )
    parser.add_argument(
        "--memory_update_prompt_version",
        type=str,
        default="auto",
        choices=["auto", "v2", "v2_1", "v2_2", "v2_3", "v2_4", "v2_5", "v3"],
        help=(
            "记忆更新 prompt 版本（默认 auto）。"
            "auto 表示与 memory_version 对齐；"
            "v2_1 仅适用于 memory_version=v2，会启用 patch 式更新：统计量代码更新，边界/要求字段按证据定点修改。"
            "v2_2 仅适用于 memory_version=v2，会进一步引入结构化 evidence bundle 与更硬的程序侧 gate。"
            "v2_3 仅适用于 memory_version=v2，会保留 v2.1 的原始 turn 级证据，只增加轻量边界摘要与轻量 gate。"
            "v2_4 仅适用于 memory_version=v2，基于 v2.1 做轻量边界保护和泛化 requirement 过滤。"
            "v2_5 仅适用于 memory_version=v2，基于 v2.1 冻结非 oracle scoring_style，并阻尼非 oracle 均值上移。"
        ),
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
        "--limit_users",
        type=int,
        default=0,
        help="最多处理的用户数量（<=0 表示不限；按用户子集截取，优先于 block limit 使用）",
    )
    parser.add_argument(
        "--user_offset",
        type=int,
        default=0,
        help="按用户子集截取时的起始偏移（默认 0，即从第一个用户开始）",
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
    # ── vLLM 支持 ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--vllm_base_url",
        type=str,
        default="",
        help=(
            "vLLM 服务地址（如 http://localhost:8000/v1）。"
            "设置后自动切换为 vLLM 模式，使用 guided_json 结构化输出。"
            "留空时使用 api_config.json 中的默认 API。"
        ),
    )
    parser.add_argument(
        "--vllm_api_key",
        type=str,
        default="EMPTY",
        help="vLLM API key（默认 EMPTY，vLLM 不校验）",
    )
    parser.add_argument(
        "--memory_vllm_base_url",
        type=str,
        default="",
        help=(
            "memory_model 使用的 OpenAI-compatible/vLLM 地址；留空时若 memory_model 与 "
            "--model 相同则沿用 --vllm_base_url，否则使用默认 API client。"
        ),
    )
    parser.add_argument(
        "--memory_vllm_api_key",
        type=str,
        default="EMPTY",
        help="memory_model vLLM API key（默认 EMPTY）。",
    )
    # ── Anchor few-shot（对现有 rubric 的补强）───────────────────────────────
    parser.add_argument(
        "--n_anchors",
        type=int,
        default=0,
        help=(
            "每轮评估时从该用户历史中检索并插入 prompt 的 anchor turns 数量。"
            "0 表示关闭（保持原 rubric-only 行为）；典型值 2-4。"
            "仅在 with_memory=True 时生效。"
        ),
    )
    parser.add_argument(
        "--turn_eval_prompt_version",
        type=str,
        default="v2",
        choices=[
            "v2",
            "v3",
            "v3_1",
            "v3_two_stage",
            "v3_two_stage_v2",
            "history_prior_delta",
            "history_prior_delta_v2",
            "history_prior_delta_v3",
            "history_prior_delta_v3_1",
            "history_prior_delta_v3_episodic",
            "history_prior_delta_v3_episodic_twopass",
            "qwen_short",
            "boundary_34",
            "boundary_34_refute",
            "boundary_34_refute_v2",
            "boundary_34_selective_refute",
            "boundary_34_selective_refute_v2",
            "boundary_34_selective_refute_v2_fullscale",
            "boundary_34_selective_refute_v3",
            "boundary_34_selective_refute_v4",
        ],
        help=(
            "turn evaluation prompt 版本。"
            "v2 为原始 rubric prompt；qwen_short 为面向 Qwen3-8B 的短 checklist prompt；"
            "v3 为 memory v3 配套 prompt，会分离 calibration 与 boundary 规则，并在证据不足时弱化边界总结；"
            "v3_1 为 memory v3 的强化版，会重新加硬 3/4 最低满意线，避免因证据不足而默认偏 SAT；"
            "v3_two_stage 为 memory v3 的两阶段版本：先判是否通过 SAT gate，再做 4/5 或 1/2/3 细分；"
            "v3_two_stage_v2 为改进版两阶段：第一层改用 selective gate + 可选复核，第二层保持 4/5 与 1/2/3 细分；"
            "history_prior_delta 显式使用 history prior，先判 residual delta 和 3/4 boundary，再由代码重建最终 1-5；"
            "history_prior_delta_v2 为 soft reconstruction 版本，仅在高置信 residual/boundary 下移动或约束分数；"
            "history_prior_delta_v3 为 hybrid vote 版本，用多个 DSAT 信号触发降到 3，同时保持 prior exact-score anchor；"
            "history_prior_delta_v3_1 为 v3 收紧版，仅在三个 DSAT 信号同时成立时触发降到 3；"
            "history_prior_delta_v3_episodic 为 v3 + 边界成对 episodic anchors，使用历史真实轮次辅助 3/4 判断；"
            "history_prior_delta_v3_episodic_twopass 为 v3.1 first-pass，仅不确定样本用 episodic anchors 二次复核；"
            "boundary_34 仅围绕 3/4 满意边界判断，并只输出 3 或 4；"
            "boundary_34_refute 会先做反证检查，再决定是否给 4；"
            "boundary_34_refute_v2 为更温和的 refute 版本，只在存在明确致命缺陷时判 3；"
            "boundary_34_selective_refute 先做温和初判，只对边界样本触发第二遍 refute；"
            "boundary_34_selective_refute_v2 会进一步收紧触发条件，并让第二遍默认维持初判；"
            "boundary_34_selective_refute_v2_fullscale 先用 v2 做 3/4 边界路由，再细化到 4/5 或 1/2/3，最终输出完整 1-5；"
            "boundary_34_selective_refute_v3 仅优化 first-pass 的 3/4 边界措辞，其余机制保持 v2；"
            "boundary_34_selective_refute_v4 以更平衡的 first-pass 同时比较最强的 3/4 证据。"
        ),
    )
    return parser


def main() -> None:
    global client, memory_client, _is_vllm

    parser = parse_args()
    args = parser.parse_args()

    # ── vLLM client 初始化 ────────────────────────────────────────────────────
    if args.vllm_base_url:
        client = OpenAI(base_url=args.vllm_base_url, api_key=args.vllm_api_key)
        _is_vllm = True
        logger.info(f"vLLM mode: base_url={args.vllm_base_url}")
    memory_model_name = args.memory_model or args.model
    if args.memory_vllm_base_url:
        memory_base_url = args.memory_vllm_base_url
        memory_api_key = args.memory_vllm_api_key
    elif memory_model_name == args.model:
        memory_base_url = args.vllm_base_url
        memory_api_key = args.vllm_api_key
    else:
        memory_base_url = ""
        memory_api_key = ""
    if memory_base_url:
        memory_client = OpenAI(
            base_url=memory_base_url,
            api_key=memory_api_key,
        )
        logger.info(f"Memory vLLM/API mode: base_url={memory_base_url}")

    with_memory = not args.no_memory
    if (
        with_memory
        and args.turn_eval_prompt_version in {
            "history_prior_delta_v3_episodic",
            "history_prior_delta_v3_episodic_twopass",
        }
        and args.n_anchors <= 0
    ):
        args.n_anchors = 4
        logger.info(
            f"{args.turn_eval_prompt_version} requires anchors; defaulting n_anchors to 4."
        )

    # 自动生成输出路径
    if not args.output_jsonl:
        model_tag = args.model.replace("/", "_").replace(":", "_")
        mode_tag = "no_memory" if not with_memory else args.memory_update_mode
        memory_tag = (
            f"_mem{args.memory_version}"
            if with_memory and args.memory_version != "v2"
            else ""
        )
        update_tag = (
            f"_upd{args.memory_update_prompt_version}"
            if with_memory and args.memory_update_mode != "none"
            and args.memory_update_prompt_version not in {"auto", "v2"}
            else ""
        )
        anchor_tag = f"_anchor{args.n_anchors}" if args.n_anchors > 0 else ""
        prompt_tag = (
            f"_{args.turn_eval_prompt_version}"
            if args.turn_eval_prompt_version != "v2" else ""
        )
        memory_model_tag = ""
        if with_memory and args.memory_model and args.memory_model != args.model:
            memory_model_tag = (
                "_memmodel"
                + args.memory_model.replace("/", "_").replace(":", "_")
            )
        args.output_jsonl = (
            f"outputs/personalized/{model_tag}_{args.split}_{mode_tag}"
            f"{memory_tag}{update_tag}{anchor_tag}{prompt_tag}{memory_model_tag}.jsonl"
        )

    logger.info(f"Model:              {args.model}")
    if with_memory:
        logger.info(f"Memory model:       {args.memory_model or args.model}")
    logger.info(f"Backend:            {'vLLM @ ' + args.vllm_base_url if _is_vllm else 'OpenAI API'}")
    if with_memory:
        memory_backend = (
            f"custom @ {memory_base_url}"
            if memory_base_url
            else "default OpenAI/API client"
        )
        logger.info(f"Memory backend:     {memory_backend}")
    logger.info(f"Split:              {args.split} (train_ratio={args.train_ratio})")
    logger.info(f"With memory:        {with_memory}")
    if with_memory:
        logger.info(f"Memory update mode: {args.memory_update_mode}")
        logger.info(f"Memory version:     {args.memory_version}")
        logger.info(f"Memory update ver:  {args.memory_update_prompt_version}")
    logger.info(f"History window:     {args.history_window_size} turns")
    logger.info(f"Anchors per turn:   {args.n_anchors}")
    logger.info(f"Turn eval prompt:   {args.turn_eval_prompt_version}")
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

    if args.limit_users > 0:
        users_in_order = list(dict.fromkeys(s.user for s in samples))
        start = max(args.user_offset, 0)
        end = start + args.limit_users
        selected_users = users_in_order[start:end]
        selected_user_set = set(selected_users)
        samples = [s for s in samples if s.user in selected_user_set]
        logger.info(
            f"Limiting to {len(selected_users)} users (offset={start}), "
            f"resulting in {len(samples)} blocks."
        )
        logger.info(f"Selected users: {selected_users}")

    if args.limit > 0:
        samples = samples[: args.limit]
        logger.info(f"Limiting to {len(samples)} blocks for debugging.")

    os.makedirs(args.memory_cache_dir, exist_ok=True)

    collect_all(
        samples=samples,
        model=args.model,
        memory_update_mode=args.memory_update_mode,
        memory_version=args.memory_version,
        memory_update_prompt_version=args.memory_update_prompt_version,
        history_window_size=args.history_window_size,
        output_jsonl=args.output_jsonl,
        max_workers=args.max_workers,
        save_memory_snapshots=args.save_memory_snapshots,
        memory_cache_dir=args.memory_cache_dir,
        with_memory=with_memory,
        n_anchors=args.n_anchors,
        turn_eval_prompt_version=args.turn_eval_prompt_version,
        memory_model=args.memory_model or None,
    )

    logger.info(f"Done. Results saved to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
