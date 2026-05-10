"""Structured prediction API call for personalized turn evaluation."""

from __future__ import annotations

import json
import os
from typing import Callable

from loguru import logger
from pydantic import BaseModel
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_fixed

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
from trace.structured_output import StructuredOutputError

StructuredParseFn = Callable[..., BaseModel]

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    before_sleep=before_sleep_log(logger, log_level=40),
)
def call_predict_turn(
    prompt: str,
    model: str,
    parse_fn: StructuredParseFn,
    raw_parse_fn: StructuredParseFn,
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
    elif prompt_version == "history_prior_delta":
        response_model = HistoryPriorDeltaPrediction
        is_boundary_prompt = True
    elif prompt_version == "history_prior_delta_v2":
        response_model = HistoryPriorDeltaV2Prediction
        is_boundary_prompt = False
    elif prompt_version == "history_prior_delta_v3":
        response_model = HistoryPriorDeltaV2Prediction
        is_boundary_prompt = False
    elif prompt_version == "history_prior_delta_v3_1":
        response_model = HistoryPriorDeltaV2Prediction
        is_boundary_prompt = False
    elif prompt_version == "history_prior_delta_v3_episodic":
        response_model = HistoryPriorDeltaV2Prediction
        is_boundary_prompt = False
    elif prompt_version == "history_prior_delta_v3_episodic_twopass":
        response_model = HistoryPriorDeltaV2Prediction
        is_boundary_prompt = False
    elif prompt_version == "history_prior_delta_v3_episodic_refine":
        response_model = EpisodicBoundaryRefinementPrediction
        is_boundary_prompt = False
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
        0.25 if prompt_version == "history_prior_delta" else
        0.25 if prompt_version == "history_prior_delta_v2" else
        0.25 if prompt_version == "history_prior_delta_v3" else
        0.25 if prompt_version == "history_prior_delta_v3_1" else
        0.25 if prompt_version == "history_prior_delta_v3_episodic" else
        0.25 if prompt_version == "history_prior_delta_v3_episodic_twopass" else
        0.2 if prompt_version == "history_prior_delta_v3_episodic_refine" else
        0.3 if prompt_version == "boundary_34" else
        0.6
    )

    try:
        if is_boundary_prompt:
            return raw_parse_fn(
                prompt,
                model,
                response_model,
                temperature=temperature,
                timeout=60,
                system_msg="You are a skilled conversational analyst.",
            )
        return parse_fn(
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
