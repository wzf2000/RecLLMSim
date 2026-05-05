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
import re
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

from openai import OpenAI

from lib.anchor_retrieval import AnchorRetriever, AnchorTurn
from lib.llm import client as _default_client
from lib.memory import (
    MemoryUpdatePatchV2_1,
    UserMemory,
    UserMemoryContent,
    UserMemoryContentV3,
    UserMemoryV3,
    build_memory_prompt,
    build_memory_prompt_v3,
    build_memory_update_prompt,
    build_memory_update_prompt_v2_1,
    build_memory_update_prompt_v3,
    build_turn_eval_fullscale_dsat_refinement_prompt,
    build_turn_eval_fullscale_sat_refinement_prompt,
    build_turn_eval_refute_followup_prompt,
    build_turn_eval_prompt,
    build_turn_eval_v3_two_stage_v2_gate_followup_prompt,
    build_turn_eval_v3_two_stage_v2_gate_prompt,
    build_turn_eval_v3_two_stage_dsat_refinement_prompt,
    build_turn_eval_v3_two_stage_gate_prompt,
    build_turn_eval_v3_two_stage_sat_refinement_prompt,
    build_turn_eval_prompt_no_memory,
    merge_memory_v2_1_patch,
)
from lib.personalized_data import (
    PersonalizedSample,
    SessionData,
    build_personalized_samples,
    dataset_stats,
)
from lib.satisfaction_constants import (
    get_reason_to_id,
    is_reason_valid_for_score,
    normalize_reason_for_score,
)

MemoryUpdateMode = Literal["none", "per_session", "per_session_oracle", "per_turn"]
MemoryVersion = Literal["v2", "v3"]
MemoryUpdatePromptVersion = Literal["auto", "v2", "v2_1", "v3"]

# ──────────────────────────────────────────────────────────────────────────────
# LLM 客户端（可在 main() 中切换为 vLLM client）
# ──────────────────────────────────────────────────────────────────────────────

client = _default_client   # module-level，可被 main() 替换为 vLLM client
_is_vllm: bool = False     # 仅用于日志标识

T = type


class StructuredOutputError(RuntimeError):
    def __init__(self, message: str, raw_text: str = "") -> None:
        super().__init__(message)
        self.raw_text = raw_text


def _message_content_to_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                parts.append(str(text))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _strip_generation_wrappers(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _normalize_quotes(text: str) -> str:
    return (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )


def _coerce_prediction_payload(payload: dict) -> dict:
    coerced = dict(payload)
    cls = coerced.get("classification")
    if isinstance(cls, str):
        cls = cls.strip()
        if cls in {"1", "2", "3", "4", "5"}:
            coerced["classification"] = int(cls)
    needs_review = coerced.get("needs_refute_review")
    if isinstance(needs_review, str):
        lowered = needs_review.strip().lower()
        if lowered in {"true", "false"}:
            coerced["needs_refute_review"] = lowered == "true"
    return coerced


def _recover_structured_output(raw_text: str, response_model: T) -> T | None:
    cleaned = _normalize_quotes(_strip_generation_wrappers(raw_text))
    if not cleaned:
        return None

    # Fast path: extract the outermost JSON object and validate directly.
    left = cleaned.find("{")
    right = cleaned.rfind("}")
    if left != -1 and right != -1 and right > left:
        candidate = cleaned[left:right + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return response_model.model_validate(_coerce_prediction_payload(parsed))
            return response_model.model_validate(parsed)
        except Exception:
            pass
    else:
        candidate = cleaned

    # Fallback: tolerant field extraction for slightly malformed JSON.
    cls_match = re.search(r'"classification"\s*:\s*"?(?P<cls>[1-5])"?', candidate)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', candidate)
    analysis_match = re.search(r'"analysis"\s*:\s*"([\s\S]*?)"\s*}', candidate)
    review_match = re.search(
        r'"needs_refute_review"\s*:\s*"?(true|false)"?',
        candidate,
        flags=re.IGNORECASE,
    )

    analysis = ""
    if analysis_match:
        analysis = analysis_match.group(1).strip()
    else:
        analysis_key = '"analysis"'
        idx = candidate.find(analysis_key)
        if idx != -1:
            tail = candidate[idx + len(analysis_key):]
            colon = tail.find(":")
            if colon != -1:
                value = tail[colon + 1:].strip()
                if value.startswith('"'):
                    value = value[1:]
                value = value.replace("</think>", "").replace("<think>", "").strip()
                if value.endswith("}"):
                    value = value[:-1].rstrip()
                if value.endswith('"'):
                    value = value[:-1]
                analysis = value.strip()

    if cls_match and reason_match and analysis:
        try:
            payload = {
                "classification": int(cls_match.group("cls")),
                "reason": reason_match.group(1).strip(),
                "analysis": analysis,
            }
            if review_match:
                payload["needs_refute_review"] = review_match.group(1).lower() == "true"
            return response_model.model_validate(payload)
        except Exception:
            return None

    return None


def _structured_parse(
    prompt: str,
    model: str,
    response_model: T,
    temperature: float = 0.3,
    timeout: int = 120,
    system_msg: str = "You are an expert user behavior analyst.",
) -> T:
    """
    统一结构化输出调用。

    OpenAI API 和 vLLM >= 0.6（含 0.18.x）均支持 json_schema response_format，
    OpenAI SDK 的 .parse() 在两者上行为一致，无需分支。
    """
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format=response_model,
        timeout=timeout,
    ).choices[0].message
    if response.parsed:
        return response.parsed

    raw_text = _message_content_to_text(getattr(response, "content", ""))
    recovered = _recover_structured_output(raw_text, response_model)
    if recovered is not None:
        logger.warning(
            f"Recovered malformed structured output for {response_model.__name__} "
            f"(model={model}, temp={temperature}, raw_len={len(raw_text)})"
        )
        return recovered

    preview = raw_text[:300].replace("\n", "\\n")
    raise StructuredOutputError(
        f"Structured parse failed: {response.refusal or 'no content'}; "
        f"raw_preview={preview}",
        raw_text=raw_text,
    )


def _structured_parse_from_raw_text(
    prompt: str,
    model: str,
    response_model: T,
    temperature: float = 0.3,
    timeout: int = 120,
    system_msg: str = "You are an expert user behavior analyst.",
) -> T:
    """
    原始文本路线：
    - 不使用 SDK .parse()
    - 直接拿 message.content
    - 本地做 wrapper stripping + tolerant recovery + schema validate

    仅用于边界 prompt，避免 vLLM/Qwen 在 json_schema 模式下偶发的
    </think> 残留和半截 JSON 直接在 SDK 层抛错。
    """
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        timeout=timeout,
    ).choices[0].message

    raw_text = _message_content_to_text(getattr(response, "content", ""))
    recovered = _recover_structured_output(raw_text, response_model)
    if recovered is not None:
        return recovered

    preview = raw_text[:300].replace("\n", "\\n")
    raise StructuredOutputError(
        f"Raw structured parse failed: {response.refusal or 'no content'}; "
        f"raw_preview={preview}",
        raw_text=raw_text,
    )


# ──────────────────────────────────────────────────────────────────────────────
# LLM 响应模型
# ──────────────────────────────────────────────────────────────────────────────

class TurnPrediction(BaseModel):
    classification: int = Field(ge=1, le=5)
    reason: str
    analysis: str


class BoundaryTurnPrediction(BaseModel):
    classification: Literal[3, 4]
    reason: str
    analysis: str


class SelectiveBoundaryTurnPrediction(BaseModel):
    classification: Literal[3, 4]
    reason: str
    analysis: str
    needs_refute_review: bool = False


class SatRefinementPrediction(BaseModel):
    classification: Literal[4, 5]
    reason: str
    analysis: str


class DsatRefinementPrediction(BaseModel):
    classification: Literal[1, 2, 3]
    reason: str
    analysis: str


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


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Memory Building
# ──────────────────────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    before_sleep=before_sleep_log(logger, log_level=40),
)
def _call_build_memory(
    prompt: str,
    model: str,
    memory_version: MemoryVersion = "v2",
) -> UserMemoryContent | UserMemoryContentV3:
    try:
        return _structured_parse(
            prompt,
            model,
            UserMemoryContent if memory_version == "v2" else UserMemoryContentV3,
            temperature=0.3, timeout=120,
            system_msg="You are an expert user behavior analyst.",
        )
    except Exception as e:
        logger.error(f"Memory building failed for {model}: {e}")
        raise e


def build_user_memory(
    sample: PersonalizedSample,
    model: str,
    memory_cache_dir: str | None = None,
    memory_version: MemoryVersion = "v2",
) -> UserMemory | UserMemoryV3:
    """
    为给定样本构建用户记忆。

    若 memory_cache_dir 不为 None，则尝试从缓存加载（避免重复调用 LLM）；
    构建成功后也会写入缓存。

    缓存文件名：{user}__{target_task}__{model}.json
    """
    cache_key = f"{sample.user}__{sample.target_task}__{model.replace('/', '_')}"
    cache_file = (
        f"{cache_key}.json"
        if memory_version == "v2"
        else f"{cache_key}__{memory_version}.json"
    )
    cache_path = (
        os.path.join(memory_cache_dir, cache_file)
        if memory_cache_dir
        else None
    )

    # 尝试从缓存加载（版本不匹配时跳过，重新构建）
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            mem = UserMemory(**data) if memory_version == "v2" else UserMemoryV3(**data)
            if mem.memory_version == memory_version:
                return mem
            logger.debug(f"Cache version mismatch ({mem.memory_version}), rebuilding: {cache_path}")
        except Exception as e:
            logger.debug(f"Cache load failed ({e}), rebuilding: {cache_path}")

    if memory_version == "v2":
        prompt = build_memory_prompt(
            user_id=sample.user,
            profile=sample.profile,
            history_sessions=sample.history_sessions,
        )
        content = _call_build_memory(prompt, model, memory_version="v2")
        assert isinstance(content, UserMemoryContent)
        memory = UserMemory.from_content(
            content,
            source_tasks=sample.history_tasks,
            n_history_sessions=sample.n_history_sessions,
            n_history_turns=sum(s.assistant_turns for s in sample.history_sessions),
        )
    else:
        prompt = build_memory_prompt_v3(
            user_id=sample.user,
            profile=sample.profile,
            history_sessions=sample.history_sessions,
        )
        content = _call_build_memory(prompt, model, memory_version="v3")
        assert isinstance(content, UserMemoryContentV3)
        memory = UserMemoryV3.from_content(
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
def _call_predict_turn(
    prompt: str,
    model: str,
    prompt_version: str = "v2",
    debug_context: str = "",
) -> (
    TurnPrediction
    | BoundaryTurnPrediction
    | SelectiveBoundaryTurnPrediction
    | SatRefinementPrediction
    | DsatRefinementPrediction
):
    is_selective_prompt = prompt_version in {
        "boundary_34_selective_refute",
        "boundary_34_selective_refute_v2",
        "boundary_34_selective_refute_v3",
        "boundary_34_selective_refute_v4",
        "v3_two_stage_v2_gate",
    }
    if prompt_version == "boundary_34_selective_refute_v2_fullscale_sat_refine":
        response_model = SatRefinementPrediction
        is_boundary_prompt = True
    elif prompt_version == "boundary_34_selective_refute_v2_fullscale_dsat_refine":
        response_model = DsatRefinementPrediction
        is_boundary_prompt = True
    elif prompt_version == "v3_two_stage_sat_refine":
        response_model = SatRefinementPrediction
        is_boundary_prompt = True
    elif prompt_version == "v3_two_stage_dsat_refine":
        response_model = DsatRefinementPrediction
        is_boundary_prompt = True
    else:
        is_boundary_prompt = prompt_version in {
            "v3_two_stage_gate",
            "v3_two_stage_v2_gate",
            "boundary_34",
            "boundary_34_refute",
            "boundary_34_refute_v2",
            "boundary_34_selective_refute",
            "boundary_34_selective_refute_v2",
            "boundary_34_selective_refute_v3",
            "boundary_34_selective_refute_v4",
            "boundary_34_selective_refute_followup",
            "boundary_34_selective_refute_v2_followup",
            "v3_two_stage_v2_gate_followup",
        }
        if is_selective_prompt:
            response_model = SelectiveBoundaryTurnPrediction
        elif is_boundary_prompt:
            response_model = BoundaryTurnPrediction
        else:
            response_model = TurnPrediction
    temperature = (
        0.2 if prompt_version == "boundary_34_refute" else
        0.2 if prompt_version == "boundary_34_selective_refute_followup" else
        0.2 if prompt_version == "boundary_34_selective_refute_v2_followup" else
        0.2 if prompt_version == "v3_two_stage_gate" else
        0.2 if prompt_version == "v3_two_stage_v2_gate_followup" else
        0.25 if prompt_version == "boundary_34_selective_refute_v2_fullscale_sat_refine" else
        0.25 if prompt_version == "boundary_34_selective_refute_v2_fullscale_dsat_refine" else
        0.25 if prompt_version == "v3_two_stage_sat_refine" else
        0.25 if prompt_version == "v3_two_stage_dsat_refine" else
        0.25 if prompt_version == "boundary_34_refute_v2" else
        0.25 if prompt_version == "boundary_34_selective_refute" else
        0.25 if prompt_version == "boundary_34_selective_refute_v2" else
        0.25 if prompt_version == "boundary_34_selective_refute_v3" else
        0.25 if prompt_version == "boundary_34_selective_refute_v4" else
        0.25 if prompt_version == "v3_two_stage_v2_gate" else
        0.3 if prompt_version == "boundary_34" else
        0.6
    )

    try:
        if is_boundary_prompt:
            return _structured_parse_from_raw_text(
                prompt,
                model,
                response_model,
                temperature=temperature,
                timeout=60,
                system_msg="You are a skilled conversational analyst.",
            )
        return _structured_parse(
            prompt,
            model,
            response_model,
            temperature=temperature,
            timeout=60,
            system_msg="You are a skilled conversational analyst.",
        )
    except Exception as e:
        dump_dir = "outputs/personalized/parse_failures"
        os.makedirs(dump_dir, exist_ok=True)
        safe_context = "".join(
            c if c.isalnum() or c in {"_", "-", "."} else "_"
            for c in (debug_context or "unknown_context")
        )[:160]
        prefix = os.path.join(dump_dir, f"{safe_context}__{prompt_version}")
        meta = {
            "debug_context": debug_context,
            "model": model,
            "prompt_version": prompt_version,
            "prompt_length": len(prompt),
            "temperature": temperature,
            "response_model": response_model.__name__,
            "parse_route": "raw_text" if is_boundary_prompt else "sdk_parse",
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

        logger.error(
            "Turn prediction parse failed: "
            f"context={debug_context}, prompt_version={prompt_version}, "
            f"prompt_len={len(prompt)}, temperature={temperature}, error={e}"
        )
        raise


def _should_trigger_selective_refute(
    pred: SelectiveBoundaryTurnPrediction,
    prompt_version: str,
) -> bool:
    if not pred.needs_refute_review:
        return False

    if prompt_version == "boundary_34_selective_refute":
        return True

    reason = pred.reason.strip()
    if prompt_version == "boundary_34_selective_refute_v2":
        if pred.classification == 3:
            return reason in {"不够细致", "其它"}
        if pred.classification == 4:
            return True
        return False
    if prompt_version == "v3_two_stage_v2_gate":
        if pred.classification == 3:
            return reason in {"不够细致", "其它"}
        if pred.classification == 4:
            return True
        return False
    if prompt_version == "boundary_34_selective_refute_v3":
        if pred.classification == 3:
            return reason in {"不够细致", "其它"}
        if pred.classification == 4:
            return True
        return False
    if prompt_version == "boundary_34_selective_refute_v4":
        if pred.classification == 3:
            return reason in {"不够细致", "其它"}
        if pred.classification == 4:
            return True
        return False

    return False


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
    """统一处理单轮预测，并在 selective 版本下按需触发二次 refute。"""
    if memory is None:
        prompt = build_turn_eval_prompt_no_memory(
            profile=session.profile,
            task_context=session.task_context,
            history_window=list(history_window),
            assistant_reply=assistant_reply,
        )
        pred = _call_predict_turn(
            prompt,
            model,
            prompt_version=turn_eval_prompt_version,
            debug_context=debug_context,
        )
        pred_reason = _normalize_pred_reason(
            pred.classification,
            pred.reason.strip(),
            default_reason=default_reason,
            debug_context=debug_context,
        )
        return {
            "pred_score": pred.classification,
            "pred_reason": pred_reason,
            "analysis": pred.analysis,
        }

    first_prompt = build_turn_eval_prompt(
        memory=memory,
        profile=session.profile,
        task_context=session.task_context,
        history_window=list(history_window),
        assistant_reply=assistant_reply,
        anchor_turns=anchors,
        prompt_version=turn_eval_prompt_version,
    )
    pred = _call_predict_turn(
        first_prompt,
        model,
        prompt_version=turn_eval_prompt_version,
        debug_context=debug_context,
    )
    pred_reason = _normalize_pred_reason(
        pred.classification,
        pred.reason.strip(),
        default_reason=default_reason,
        debug_context=debug_context,
    )

    if turn_eval_prompt_version not in {
        "boundary_34_selective_refute",
        "boundary_34_selective_refute_v2",
        "boundary_34_selective_refute_v3",
        "boundary_34_selective_refute_v4",
    }:
        return {
            "pred_score": pred.classification,
            "pred_reason": pred_reason,
            "analysis": pred.analysis,
        }

    assert isinstance(pred, SelectiveBoundaryTurnPrediction)
    should_trigger = _should_trigger_selective_refute(pred, turn_eval_prompt_version)
    result = {
        "pred_score": pred.classification,
        "pred_reason": pred_reason,
        "analysis": pred.analysis,
        "analysis_first_pass": pred.analysis,
        "selective_refute_triggered": should_trigger,
        "selective_refute_applied": False,
        "selective_refute_initial_score": pred.classification,
        "selective_refute_initial_reason": pred_reason,
        "selective_refute_model_flag": pred.needs_refute_review,
    }

    if not should_trigger:
        return result

    followup_prompt = build_turn_eval_refute_followup_prompt(
        memory=memory,
        profile=session.profile,
        task_context=session.task_context,
        history_window=list(history_window),
        assistant_reply=assistant_reply,
        initial_classification=pred.classification,
        initial_reason=pred_reason,
        initial_analysis=pred.analysis,
        prompt_version=turn_eval_prompt_version,
    )

    followup_prompt_version = (
        "boundary_34_selective_refute_v2_followup"
        if turn_eval_prompt_version in {
            "boundary_34_selective_refute_v2",
            "boundary_34_selective_refute_v3",
            "boundary_34_selective_refute_v4",
        }
        else "boundary_34_selective_refute_followup"
    )
    try:
        followup = _call_predict_turn(
            followup_prompt,
            model,
            prompt_version=followup_prompt_version,
            debug_context=f"{debug_context}__refute",
        )
        assert isinstance(followup, BoundaryTurnPrediction)
        followup_reason = _normalize_pred_reason(
            followup.classification,
            followup.reason.strip(),
            default_reason=default_reason,
            debug_context=f"{debug_context}__refute",
        )
        result.update(
            {
                "pred_score": followup.classification,
                "pred_reason": followup_reason,
                "analysis": (
                    f"[first_pass] {pred.analysis}\n"
                    f"[refute] {followup.analysis}"
                ),
                "analysis_refute": followup.analysis,
                "selective_refute_applied": True,
            }
        )
    except Exception as e:
        logger.warning(
            f"Selective refute follow-up failed for {debug_context}: {e}; "
            "keeping first-pass decision."
        )
        result["analysis"] = (
            f"[first_pass] {pred.analysis}\n"
            "[refute] follow-up failed, keep first-pass decision"
        )

    return result


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
    """
    分层 1-5 pipeline：
    1. 先复用当前最稳的 boundary_34_selective_refute_v2 做 3/4 路由
    2. SAT 分支再细化到 4/5；DSAT 分支再细化到 1/2/3
    """
    if memory is None:
        prompt = build_turn_eval_prompt_no_memory(
            profile=session.profile,
            task_context=session.task_context,
            history_window=list(history_window),
            assistant_reply=assistant_reply,
        )
        pred = _call_predict_turn(
            prompt,
            model,
            prompt_version="v2",
            debug_context=debug_context,
        )
        assert isinstance(pred, TurnPrediction)
        return {
            "pred_score": pred.classification,
            "pred_reason": _normalize_pred_reason(
                pred.classification,
                pred.reason.strip(),
                default_reason=default_reason,
                debug_context=debug_context,
            ),
            "analysis": pred.analysis,
            "fullscale_router_score": None,
            "fullscale_router_reason": "",
            "fullscale_router_analysis": "",
            "fullscale_branch": "no_memory_fallback",
            "fullscale_refine_applied": False,
        }

    router_result = _predict_turn_with_optional_selective_refute(
        memory=memory,
        session=session,
        model=model,
        history_window=history_window,
        assistant_reply=assistant_reply,
        turn_eval_prompt_version="boundary_34_selective_refute_v2",
        debug_context=f"{debug_context}__router",
        default_reason=default_reason,
        anchors=anchors,
    )

    router_score = int(router_result["pred_score"])
    router_reason = router_result["pred_reason"].strip()
    router_analysis = router_result["analysis"]
    base_result = {
        "fullscale_router_score": router_score,
        "fullscale_router_reason": router_reason,
        "fullscale_router_analysis": router_analysis,
        "fullscale_router_triggered": router_result.get("selective_refute_triggered", False),
        "fullscale_router_applied": router_result.get("selective_refute_applied", False),
        "fullscale_router_initial_score": router_result.get("selective_refute_initial_score"),
        "fullscale_router_initial_reason": router_result.get("selective_refute_initial_reason"),
        "fullscale_router_model_flag": router_result.get("selective_refute_model_flag"),
        "analysis_router": router_analysis,
    }

    if router_score >= 4:
        refine_prompt = build_turn_eval_fullscale_sat_refinement_prompt(
            memory=memory,
            profile=session.profile,
            task_context=session.task_context,
            history_window=list(history_window),
            assistant_reply=assistant_reply,
            router_reason=router_reason,
            router_analysis=router_analysis,
            anchor_turns=anchors,
        )
        refine = _call_predict_turn(
            refine_prompt,
            model,
            prompt_version="boundary_34_selective_refute_v2_fullscale_sat_refine",
            debug_context=f"{debug_context}__sat_refine",
        )
        assert isinstance(refine, SatRefinementPrediction)
        final_reason = _normalize_pred_reason(
            refine.classification,
            refine.reason.strip(),
            default_reason=default_reason,
            debug_context=f"{debug_context}__sat_refine",
        )
        return {
            "pred_score": refine.classification,
            "pred_reason": final_reason,
            "analysis": f"[router] {router_analysis}\n[sat_refine] {refine.analysis}",
            "analysis_sat_refine": refine.analysis,
            "fullscale_branch": "sat_45",
            "fullscale_refine_applied": True,
            **base_result,
        }

    refine_prompt = build_turn_eval_fullscale_dsat_refinement_prompt(
        memory=memory,
        profile=session.profile,
        task_context=session.task_context,
        history_window=list(history_window),
        assistant_reply=assistant_reply,
        router_reason=router_reason,
        router_analysis=router_analysis,
        anchor_turns=anchors,
    )
    refine = _call_predict_turn(
        refine_prompt,
        model,
        prompt_version="boundary_34_selective_refute_v2_fullscale_dsat_refine",
        debug_context=f"{debug_context}__dsat_refine",
    )
    assert isinstance(refine, DsatRefinementPrediction)
    final_reason = _normalize_pred_reason(
        refine.classification,
        refine.reason.strip(),
        default_reason=default_reason,
        debug_context=f"{debug_context}__dsat_refine",
    )
    return {
        "pred_score": refine.classification,
        "pred_reason": final_reason,
        "analysis": f"[router] {router_analysis}\n[dsat_refine] {refine.analysis}",
        "analysis_dsat_refine": refine.analysis,
        "fullscale_branch": "dsat_123",
        "fullscale_refine_applied": True,
        **base_result,
    }


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
    """
    基于 memory v3 的两阶段 1-5 pipeline：
    1. Stage 1：只做 SAT gate（3/4）
    2. Stage 2：SAT 分支细化 4/5，DSAT 分支细化 1/2/3
    """
    if memory is None:
        prompt = build_turn_eval_prompt_no_memory(
            profile=session.profile,
            task_context=session.task_context,
            history_window=list(history_window),
            assistant_reply=assistant_reply,
        )
        pred = _call_predict_turn(
            prompt,
            model,
            prompt_version="v2",
            debug_context=debug_context,
        )
        assert isinstance(pred, TurnPrediction)
        return {
            "pred_score": pred.classification,
            "pred_reason": _normalize_pred_reason(
                pred.classification,
                pred.reason.strip(),
                default_reason=default_reason,
                debug_context=debug_context,
            ),
            "analysis": pred.analysis,
            "two_stage_gate_score": None,
            "two_stage_gate_reason": "",
            "two_stage_gate_analysis": "",
            "two_stage_branch": "no_memory_fallback",
            "two_stage_refine_applied": False,
        }

    gate_prompt = build_turn_eval_v3_two_stage_gate_prompt(
        memory=memory,
        profile=session.profile,
        task_context=session.task_context,
        history_window=list(history_window),
        assistant_reply=assistant_reply,
        anchor_turns=anchors,
    )
    gate_pred = _call_predict_turn(
        gate_prompt,
        model,
        prompt_version="v3_two_stage_gate",
        debug_context=f"{debug_context}__gate",
    )
    assert isinstance(gate_pred, BoundaryTurnPrediction)
    gate_reason = _normalize_pred_reason(
        gate_pred.classification,
        gate_pred.reason.strip(),
        default_reason=default_reason,
        debug_context=f"{debug_context}__gate",
    )
    gate_analysis = gate_pred.analysis
    base_result = {
        "two_stage_gate_score": gate_pred.classification,
        "two_stage_gate_reason": gate_reason,
        "two_stage_gate_analysis": gate_analysis,
        "analysis_gate": gate_analysis,
    }

    if gate_pred.classification >= 4:
        refine_prompt = build_turn_eval_v3_two_stage_sat_refinement_prompt(
            memory=memory,
            profile=session.profile,
            task_context=session.task_context,
            history_window=list(history_window),
            assistant_reply=assistant_reply,
            gate_reason=gate_reason,
            gate_analysis=gate_analysis,
            anchor_turns=anchors,
        )
        refine = _call_predict_turn(
            refine_prompt,
            model,
            prompt_version="v3_two_stage_sat_refine",
            debug_context=f"{debug_context}__sat_refine",
        )
        assert isinstance(refine, SatRefinementPrediction)
        final_reason = _normalize_pred_reason(
            refine.classification,
            refine.reason.strip(),
            default_reason=default_reason,
            debug_context=f"{debug_context}__sat_refine",
        )
        return {
            "pred_score": refine.classification,
            "pred_reason": final_reason,
            "analysis": f"[gate] {gate_analysis}\n[sat_refine] {refine.analysis}",
            "analysis_sat_refine": refine.analysis,
            "two_stage_branch": "sat_45",
            "two_stage_refine_applied": True,
            **base_result,
        }

    refine_prompt = build_turn_eval_v3_two_stage_dsat_refinement_prompt(
        memory=memory,
        profile=session.profile,
        task_context=session.task_context,
        history_window=list(history_window),
        assistant_reply=assistant_reply,
        gate_reason=gate_reason,
        gate_analysis=gate_analysis,
        anchor_turns=anchors,
    )
    refine = _call_predict_turn(
        refine_prompt,
        model,
        prompt_version="v3_two_stage_dsat_refine",
        debug_context=f"{debug_context}__dsat_refine",
    )
    assert isinstance(refine, DsatRefinementPrediction)
    final_reason = _normalize_pred_reason(
        refine.classification,
        refine.reason.strip(),
        default_reason=default_reason,
        debug_context=f"{debug_context}__dsat_refine",
    )
    return {
        "pred_score": refine.classification,
        "pred_reason": final_reason,
        "analysis": f"[gate] {gate_analysis}\n[dsat_refine] {refine.analysis}",
        "analysis_dsat_refine": refine.analysis,
        "two_stage_branch": "dsat_123",
        "two_stage_refine_applied": True,
        **base_result,
    }


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
    """
    memory v3 两阶段 v2：
    1. 第一层改为 selective-refute 风格的 SAT gate
    2. 第二层继续沿用现有 4/5 与 1/2/3 refine
    """
    if memory is None:
        return _predict_turn_v3_two_stage(
            memory=memory,
            session=session,
            model=model,
            history_window=history_window,
            assistant_reply=assistant_reply,
            debug_context=debug_context,
            default_reason=default_reason,
            anchors=anchors,
        )

    gate_prompt = build_turn_eval_v3_two_stage_v2_gate_prompt(
        memory=memory,
        profile=session.profile,
        task_context=session.task_context,
        history_window=list(history_window),
        assistant_reply=assistant_reply,
        anchor_turns=anchors,
    )
    gate_pred = _call_predict_turn(
        gate_prompt,
        model,
        prompt_version="v3_two_stage_v2_gate",
        debug_context=f"{debug_context}__gate",
    )
    assert isinstance(gate_pred, SelectiveBoundaryTurnPrediction)
    gate_reason = _normalize_pred_reason(
        gate_pred.classification,
        gate_pred.reason.strip(),
        default_reason=default_reason,
        debug_context=f"{debug_context}__gate",
    )
    should_trigger_gate_refute = _should_trigger_selective_refute(
        gate_pred,
        "v3_two_stage_v2_gate",
    )
    gate_score = gate_pred.classification
    gate_analysis = gate_pred.analysis
    gate_reason_final = gate_reason
    gate_followup_analysis = ""

    if should_trigger_gate_refute:
        followup_prompt = build_turn_eval_v3_two_stage_v2_gate_followup_prompt(
            memory=memory,
            profile=session.profile,
            task_context=session.task_context,
            history_window=list(history_window),
            assistant_reply=assistant_reply,
            initial_classification=gate_pred.classification,
            initial_reason=gate_reason,
            initial_analysis=gate_pred.analysis,
        )
        followup = _call_predict_turn(
            followup_prompt,
            model,
            prompt_version="v3_two_stage_v2_gate_followup",
            debug_context=f"{debug_context}__gate_followup",
        )
        assert isinstance(followup, BoundaryTurnPrediction)
        gate_score = followup.classification
        gate_reason_final = _normalize_pred_reason(
            followup.classification,
            followup.reason.strip(),
            default_reason=default_reason,
            debug_context=f"{debug_context}__gate_followup",
        )
        gate_followup_analysis = followup.analysis
        gate_analysis = f"[first_pass] {gate_pred.analysis}\n[gate_followup] {followup.analysis}"

    base_result = {
        "two_stage_gate_score": gate_score,
        "two_stage_gate_reason": gate_reason_final,
        "two_stage_gate_analysis": gate_analysis,
        "analysis_gate": gate_analysis,
        "two_stage_gate_model_flag": gate_pred.needs_refute_review,
        "two_stage_gate_triggered": should_trigger_gate_refute,
        "two_stage_gate_refute_applied": should_trigger_gate_refute,
        "analysis_gate_first_pass": gate_pred.analysis,
        "analysis_gate_followup": gate_followup_analysis,
    }

    if gate_score >= 4:
        refine_prompt = build_turn_eval_v3_two_stage_sat_refinement_prompt(
            memory=memory,
            profile=session.profile,
            task_context=session.task_context,
            history_window=list(history_window),
            assistant_reply=assistant_reply,
            gate_reason=gate_reason_final,
            gate_analysis=gate_analysis,
            anchor_turns=anchors,
        )
        refine = _call_predict_turn(
            refine_prompt,
            model,
            prompt_version="v3_two_stage_sat_refine",
            debug_context=f"{debug_context}__sat_refine",
        )
        assert isinstance(refine, SatRefinementPrediction)
        final_reason = _normalize_pred_reason(
            refine.classification,
            refine.reason.strip(),
            default_reason=default_reason,
            debug_context=f"{debug_context}__sat_refine",
        )
        return {
            "pred_score": refine.classification,
            "pred_reason": final_reason,
            "analysis": f"[gate] {gate_analysis}\n[sat_refine] {refine.analysis}",
            "analysis_sat_refine": refine.analysis,
            "two_stage_branch": "sat_45",
            "two_stage_refine_applied": True,
            **base_result,
        }

    refine_prompt = build_turn_eval_v3_two_stage_dsat_refinement_prompt(
        memory=memory,
        profile=session.profile,
        task_context=session.task_context,
        history_window=list(history_window),
        assistant_reply=assistant_reply,
        gate_reason=gate_reason_final,
        gate_analysis=gate_analysis,
        anchor_turns=anchors,
    )
    refine = _call_predict_turn(
        refine_prompt,
        model,
        prompt_version="v3_two_stage_dsat_refine",
        debug_context=f"{debug_context}__dsat_refine",
    )
    assert isinstance(refine, DsatRefinementPrediction)
    final_reason = _normalize_pred_reason(
        refine.classification,
        refine.reason.strip(),
        default_reason=default_reason,
        debug_context=f"{debug_context}__dsat_refine",
    )
    return {
        "pred_score": refine.classification,
        "pred_reason": final_reason,
        "analysis": f"[gate] {gate_analysis}\n[dsat_refine] {refine.analysis}",
        "analysis_dsat_refine": refine.analysis,
        "two_stage_branch": "dsat_123",
        "two_stage_refine_applied": True,
        **base_result,
    }


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
    last_user_msg: str = ""

    for utt in session.history:
        if utt["role"] == "user":
            last_user_msg = utt["content"]
        if utt["role"] == "assistant":
            # 检索 anchor turns（若启用）
            anchors: list[AnchorTurn] | None = None
            if memory is not None and retriever is not None and n_anchors > 0:
                anchors = retriever.retrieve(
                    query_user_msg=last_user_msg,
                    query_assistant_reply=utt["content"],
                    k=n_anchors,
                )

            debug_context = (
                f"{block_id}__{os.path.basename(session.file_path)}__turn_{assistant_turn_idx}"
                if block_id else
                f"{os.path.basename(session.file_path)}__turn_{assistant_turn_idx}"
            )
            if turn_eval_prompt_version == "boundary_34_selective_refute_v2_fullscale":
                pred_result = _predict_turn_fullscale_from_boundary_v2(
                    memory=memory,
                    session=session,
                    model=model,
                    history_window=history_window,
                    assistant_reply=utt["content"],
                    debug_context=debug_context,
                    default_reason=default_reason,
                    anchors=anchors,
                )
            elif turn_eval_prompt_version == "v3_two_stage":
                pred_result = _predict_turn_v3_two_stage(
                    memory=memory,
                    session=session,
                    model=model,
                    history_window=history_window,
                    assistant_reply=utt["content"],
                    debug_context=debug_context,
                    default_reason=default_reason,
                    anchors=anchors,
                )
            elif turn_eval_prompt_version == "v3_two_stage_v2":
                pred_result = _predict_turn_v3_two_stage_v2(
                    memory=memory,
                    session=session,
                    model=model,
                    history_window=history_window,
                    assistant_reply=utt["content"],
                    debug_context=debug_context,
                    default_reason=default_reason,
                    anchors=anchors,
                )
            else:
                pred_result = _predict_turn_with_optional_selective_refute(
                    memory=memory,
                    session=session,
                    model=model,
                    history_window=history_window,
                    assistant_reply=utt["content"],
                    turn_eval_prompt_version=turn_eval_prompt_version,
                    debug_context=debug_context,
                    default_reason=default_reason,
                    anchors=anchors,
                )
            pred_reason = pred_result["pred_reason"].strip()
            if pred_reason not in valid_reasons:
                pred_reason = default_reason

            gold_score = session.satisfaction_scores[assistant_turn_idx]
            gold_reason = session.dissatisfaction_reasons[assistant_turn_idx]

            turn_result = {
                "turn_idx": assistant_turn_idx,
                "pred_score": pred_result["pred_score"],
                "pred_reason": pred_reason,
                "gold_score": gold_score,
                "gold_reason": gold_reason,
                "analysis": pred_result["analysis"],
            }
            for optional_key in (
                "analysis_first_pass",
                "analysis_refute",
                "selective_refute_triggered",
                "selective_refute_applied",
                "selective_refute_initial_score",
                "selective_refute_initial_reason",
                "selective_refute_model_flag",
                "analysis_router",
                "analysis_sat_refine",
                "analysis_dsat_refine",
                "fullscale_router_score",
                "fullscale_router_reason",
                "fullscale_router_analysis",
                "fullscale_router_triggered",
                "fullscale_router_applied",
                "fullscale_router_initial_score",
                "fullscale_router_initial_reason",
                "fullscale_router_model_flag",
                "fullscale_branch",
                "fullscale_refine_applied",
                "analysis_gate",
                "two_stage_gate_score",
                "two_stage_gate_reason",
                "two_stage_gate_analysis",
                "two_stage_branch",
                "two_stage_refine_applied",
                "two_stage_gate_model_flag",
                "two_stage_gate_triggered",
                "two_stage_gate_refute_applied",
                "analysis_gate_first_pass",
                "analysis_gate_followup",
            ):
                if optional_key in pred_result:
                    turn_result[optional_key] = pred_result[optional_key]
            results.append(turn_result)
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
def _call_update_memory(
    prompt: str,
    model: str,
    update_prompt_version: MemoryUpdatePromptVersion = "v2",
) -> UserMemoryContent | UserMemoryContentV3 | MemoryUpdatePatchV2_1:
    response_model = (
        MemoryUpdatePatchV2_1 if update_prompt_version == "v2_1"
        else UserMemoryContent if update_prompt_version == "v2"
        else UserMemoryContentV3
    )
    return _structured_parse(
        prompt,
        model,
        response_model,
        temperature=0.3, timeout=120,
        system_msg="You are an expert user behavior analyst.",
    )


def _resolve_memory_update_prompt_version(
    memory_version: MemoryVersion,
    memory_update_prompt_version: MemoryUpdatePromptVersion,
) -> Literal["v2", "v2_1", "v3"]:
    if memory_update_prompt_version == "auto":
        return "v2" if memory_version == "v2" else "v3"
    if memory_version == "v2" and memory_update_prompt_version in {"v2", "v2_1"}:
        return memory_update_prompt_version
    if memory_version == "v3" and memory_update_prompt_version == "v3":
        return "v3"
    raise ValueError(
        f"Incompatible memory update prompt version: memory_version={memory_version}, "
        f"memory_update_prompt_version={memory_update_prompt_version}"
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
    """在预测完一个 session 后更新用户记忆。"""
    resolved_update_version = _resolve_memory_update_prompt_version(
        memory_version,
        memory_update_prompt_version,
    )
    if memory_version == "v2":
        assert isinstance(memory, UserMemory)
        if resolved_update_version == "v2_1":
            prompt = build_memory_update_prompt_v2_1(
                existing_memory=memory,
                new_session=session,
                turn_predictions=turn_predictions,
                use_oracle_labels=use_oracle_labels,
            )
            patch = _call_update_memory(prompt, model, update_prompt_version="v2_1")
            assert isinstance(patch, MemoryUpdatePatchV2_1)
            return merge_memory_v2_1_patch(
                existing_memory=memory,
                patch=patch,
                turn_predictions=turn_predictions,
                use_oracle_labels=use_oracle_labels,
            )

        prompt = build_memory_update_prompt(
            existing_memory=memory,
            new_session=session,
            turn_predictions=turn_predictions,
            use_oracle_labels=use_oracle_labels,
        )
        content = _call_update_memory(prompt, model, update_prompt_version="v2")
        assert isinstance(content, UserMemoryContent)
        return UserMemory.from_content(
            content,
            source_tasks=memory.source_tasks,
            n_history_sessions=memory.n_history_sessions + 1,
            n_history_turns=memory.n_history_turns + len(turn_predictions),
        )

    assert isinstance(memory, UserMemoryV3)
    prompt = build_memory_update_prompt_v3(
        existing_memory=memory,
        new_session=session,
        turn_predictions=turn_predictions,
        use_oracle_labels=use_oracle_labels,
    )
    content = _call_update_memory(prompt, model, update_prompt_version="v3")
    assert isinstance(content, UserMemoryContentV3)
    return UserMemoryV3.from_content(
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
    memory_version: MemoryVersion = "v2",
    memory_update_prompt_version: MemoryUpdatePromptVersion = "auto",
    history_window_size: int = 5,
    save_memory_snapshots: bool = False,
    memory_cache_dir: str | None = None,
    with_memory: bool = True,
    n_anchors: int = 0,
    turn_eval_prompt_version: str = "v2",
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
        build_user_memory(
            sample,
            model,
            memory_cache_dir=memory_cache_dir,
            memory_version=memory_version,
        )
        if with_memory
        else None
    )

    # Anchor retriever（每个 sample 构建一次，复用 history_sessions）
    retriever: AnchorRetriever | None = None
    if with_memory and n_anchors > 0:
        retriever = AnchorRetriever(sample.history_sessions)

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
                retriever=retriever,
                n_anchors=n_anchors,
                turn_eval_prompt_version=turn_eval_prompt_version,
                block_id=sample.block_id,
                memory_version=memory_version,
                memory_update_prompt_version=memory_update_prompt_version,
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
                retriever=retriever,
                n_anchors=n_anchors,
                turn_eval_prompt_version=turn_eval_prompt_version,
                block_id=sample.block_id,
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
                "memory_version": memory.memory_version if memory is not None else "none",
                "memory_update_prompt_version": memory_update_prompt_version if with_memory else "none",
                "turn_eval_prompt_version": turn_eval_prompt_version,
            }
            for optional_key in (
                "analysis_first_pass",
                "analysis_refute",
                "selective_refute_triggered",
                "selective_refute_applied",
                "selective_refute_initial_score",
                "selective_refute_initial_reason",
                "selective_refute_model_flag",
                "analysis_router",
                "analysis_sat_refine",
                "analysis_dsat_refine",
                "fullscale_router_score",
                "fullscale_router_reason",
                "fullscale_router_analysis",
                "fullscale_router_triggered",
                "fullscale_router_applied",
                "fullscale_router_initial_score",
                "fullscale_router_initial_reason",
                "fullscale_router_model_flag",
                "fullscale_branch",
                "fullscale_refine_applied",
                "analysis_gate",
                "two_stage_gate_score",
                "two_stage_gate_reason",
                "two_stage_gate_analysis",
                "two_stage_branch",
                "two_stage_refine_applied",
                "two_stage_gate_model_flag",
                "two_stage_gate_triggered",
                "two_stage_gate_refute_applied",
                "analysis_gate_first_pass",
                "analysis_gate_followup",
            ):
                if optional_key in r:
                    record[optional_key] = r[optional_key]
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
                    memory_version=memory_version,
                    memory_update_prompt_version=memory_update_prompt_version,
                )
            except Exception as e:
                logger.warning(
                    f"Memory update failed for {sample.user}/{session_file}: {e}, "
                    f"keeping existing memory."
                )

    return all_turn_records


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
) -> list[dict]:
    """
    per_turn 模式：每预测一轮后立即更新记忆。
    由于需要顺序执行，不能并行化。
    """
    results: list[dict] = []
    history_window: list[str] = []        # 格式化字符串，用于 eval prompt
    history_window_dicts: list[dict] = [] # 原始 dict，用于构造 mini_session
    assistant_turn_idx = 0
    last_user_msg: str = ""

    for utt in session.history:
        if utt["role"] == "user":
            last_user_msg = utt["content"]
        if utt["role"] == "assistant":
            anchors: list[AnchorTurn] | None = None
            if retriever is not None and n_anchors > 0:
                anchors = retriever.retrieve(
                    query_user_msg=last_user_msg,
                    query_assistant_reply=utt["content"],
                    k=n_anchors,
                )
            debug_context = (
                f"{block_id}__{os.path.basename(session.file_path)}__turn_{assistant_turn_idx}"
                if block_id else
                f"{os.path.basename(session.file_path)}__turn_{assistant_turn_idx}"
            )
            if turn_eval_prompt_version == "boundary_34_selective_refute_v2_fullscale":
                pred_result = _predict_turn_fullscale_from_boundary_v2(
                    memory=memory,
                    session=session,
                    model=model,
                    history_window=history_window,
                    assistant_reply=utt["content"],
                    debug_context=debug_context,
                    default_reason=default_reason,
                    anchors=anchors,
                )
            elif turn_eval_prompt_version == "v3_two_stage":
                pred_result = _predict_turn_v3_two_stage(
                    memory=memory,
                    session=session,
                    model=model,
                    history_window=history_window,
                    assistant_reply=utt["content"],
                    debug_context=debug_context,
                    default_reason=default_reason,
                    anchors=anchors,
                )
            elif turn_eval_prompt_version == "v3_two_stage_v2":
                pred_result = _predict_turn_v3_two_stage_v2(
                    memory=memory,
                    session=session,
                    model=model,
                    history_window=history_window,
                    assistant_reply=utt["content"],
                    debug_context=debug_context,
                    default_reason=default_reason,
                    anchors=anchors,
                )
            else:
                pred_result = _predict_turn_with_optional_selective_refute(
                    memory=memory,
                    session=session,
                    model=model,
                    history_window=history_window,
                    assistant_reply=utt["content"],
                    turn_eval_prompt_version=turn_eval_prompt_version,
                    debug_context=debug_context,
                    default_reason=default_reason,
                    anchors=anchors,
                )
            pred_reason = pred_result["pred_reason"].strip()
            if pred_reason not in valid_reasons:
                pred_reason = default_reason

            gold_score = session.satisfaction_scores[assistant_turn_idx]
            gold_reason = session.dissatisfaction_reasons[assistant_turn_idx]

            turn_result = {
                "turn_idx": assistant_turn_idx,
                "pred_score": pred_result["pred_score"],
                "pred_reason": pred_reason,
                "gold_score": gold_score,
                "gold_reason": gold_reason,
                "analysis": pred_result["analysis"],
            }
            for optional_key in (
                "analysis_first_pass",
                "analysis_refute",
                "selective_refute_triggered",
                "selective_refute_applied",
                "selective_refute_initial_score",
                "selective_refute_initial_reason",
                "selective_refute_model_flag",
                "analysis_router",
                "analysis_sat_refine",
                "analysis_dsat_refine",
                "fullscale_router_score",
                "fullscale_router_reason",
                "fullscale_router_analysis",
                "fullscale_router_triggered",
                "fullscale_router_applied",
                "fullscale_router_initial_score",
                "fullscale_router_initial_reason",
                "fullscale_router_model_flag",
                "fullscale_branch",
                "fullscale_refine_applied",
                "analysis_gate",
                "two_stage_gate_score",
                "two_stage_gate_reason",
                "two_stage_gate_analysis",
                "two_stage_branch",
                "two_stage_refine_applied",
                "two_stage_gate_model_flag",
                "two_stage_gate_triggered",
                "two_stage_gate_refute_applied",
                "analysis_gate_first_pass",
                "analysis_gate_followup",
            ):
                if optional_key in pred_result:
                    turn_result[optional_key] = pred_result[optional_key]
            results.append(turn_result)

            # 逐轮更新记忆：用原始 dict 列表构造 mini_session
            mini_session = SessionData(
                user=session.user,
                task=session.task,
                file_path=session.file_path,
                task_context=session.task_context,
                profile=session.profile,
                history=list(history_window_dicts) + [utt],
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
                    memory_version=memory_version,
                    memory_update_prompt_version=memory_update_prompt_version,
                )
            except Exception as e:
                logger.warning(f"Per-turn memory update failed at turn {assistant_turn_idx}: {e}")

            assistant_turn_idx += 1

        role_label = "用户" if utt["role"] == "user" else "助手"
        history_window.append(f"{role_label}：{utt['content']}")
        history_window_dicts.append(utt)
        while len(history_window) > history_window_size:
            history_window.pop(0)
        while len(history_window_dicts) > history_window_size:
            history_window_dicts.pop(0)

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
            memory_version=memory_version,
            memory_update_prompt_version=memory_update_prompt_version,
            history_window_size=history_window_size,
            save_memory_snapshots=save_memory_snapshots,
            memory_cache_dir=memory_cache_dir,
            with_memory=with_memory,
            n_anchors=n_anchors,
            turn_eval_prompt_version=turn_eval_prompt_version,
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
        choices=["auto", "v2", "v2_1", "v3"],
        help=(
            "记忆更新 prompt 版本（默认 auto）。"
            "auto 表示与 memory_version 对齐；"
            "v2_1 仅适用于 memory_version=v2，会启用 patch 式更新：统计量代码更新，边界/要求字段按证据定点修改。"
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
    global client, _is_vllm

    parser = parse_args()
    args = parser.parse_args()

    # ── vLLM client 初始化 ────────────────────────────────────────────────────
    if args.vllm_base_url:
        client = OpenAI(base_url=args.vllm_base_url, api_key=args.vllm_api_key)
        _is_vllm = True
        logger.info(f"vLLM mode: base_url={args.vllm_base_url}")

    with_memory = not args.no_memory

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
        args.output_jsonl = (
            f"outputs/personalized/{model_tag}_{args.split}_{mode_tag}{memory_tag}{update_tag}{anchor_tag}{prompt_tag}.jsonl"
        )

    logger.info(f"Model:              {args.model}")
    logger.info(f"Backend:            {'vLLM @ ' + args.vllm_base_url if _is_vllm else 'OpenAI API'}")
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
    )

    logger.info(f"Done. Results saved to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
