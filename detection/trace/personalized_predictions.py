"""Prediction schemas and score reconstruction helpers for personalized traces."""

from __future__ import annotations

from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field

from lib.anchor_retrieval import AnchorRetriever, AnchorTurn
from lib.satisfaction_constants import (
    is_reason_valid_for_score,
    normalize_reason_for_score,
)


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


class HistoryPriorDeltaPrediction(BaseModel):
    classification: int = Field(ge=1, le=5)
    reason: str
    analysis: str
    history_prior_score: float = Field(ge=1, le=5)
    delta_label: Literal["below", "around", "above"]
    delta_score: int = Field(ge=-2, le=2)
    passes_satisfaction_boundary: bool
    boundary_score: Literal[3, 4]


class HistoryPriorDeltaV2Prediction(BaseModel):
    classification: int = Field(ge=1, le=5)
    reason: str
    analysis: str
    history_prior_score: float = Field(ge=1, le=5)
    delta_label: Literal["below", "around", "above"]
    delta_score: int = Field(ge=-2, le=2)
    delta_confidence: Literal["low", "medium", "high"]
    passes_satisfaction_boundary: bool
    boundary_score: Literal[3, 4]
    boundary_confidence: Literal["low", "medium", "high"]
    strong_failure_evidence: bool = False
    strong_excellence_evidence: bool = False


class SatRefinementPrediction(BaseModel):
    classification: Literal[4, 5]
    reason: str
    analysis: str


class DsatRefinementPrediction(BaseModel):
    classification: Literal[1, 2, 3]
    reason: str
    analysis: str


class EpisodicBoundaryRefinementPrediction(BaseModel):
    classification: Literal[3, 4]
    reason: str
    analysis: str
    closest_evidence_side: Literal["dsat", "sat", "mixed"]
    evidence_match_confidence: Literal["low", "medium", "high"]


def normalize_pred_reason(
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


def clip_score(score: int | float) -> int:
    return max(1, min(5, int(round(score))))


def reconstruct_history_prior_delta_score(pred: HistoryPriorDeltaPrediction) -> int:
    """Rebuild the final 1-5 score from prior + delta, then enforce the 3/4 gate."""
    reconstructed = clip_score(pred.history_prior_score + pred.delta_score)
    boundary_score = 4 if pred.passes_satisfaction_boundary else 3
    if pred.boundary_score in {3, 4}:
        boundary_score = pred.boundary_score
    if boundary_score >= 4:
        return max(4, reconstructed)
    return min(3, reconstructed)


def sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def reconstruct_history_prior_delta_v2_score(pred: HistoryPriorDeltaV2Prediction) -> int:
    """
    Soft reconstruction for residual judging.

    Keep the rounded user prior as the default exact score. Use residuals only
    when the model reports enough evidence, and apply 3/4 boundary constraints
    only for high-confidence boundary decisions.
    """
    score = clip_score(pred.history_prior_score)
    delta_step = 0
    if pred.delta_confidence == "high":
        delta_step = sign(pred.delta_score)
    elif pred.delta_confidence == "medium" and abs(pred.delta_score) == 2:
        delta_step = sign(pred.delta_score)

    if delta_step:
        score += delta_step

    score = clip_score(score)
    if pred.boundary_confidence == "high":
        boundary_score = 4 if pred.passes_satisfaction_boundary else 3
        if pred.boundary_score in {3, 4}:
            boundary_score = pred.boundary_score
        if boundary_score >= 4:
            score = max(4, score)
        else:
            score = min(3, score)
    return clip_score(score)


def history_prior_delta_v3_dsat_votes(pred: HistoryPriorDeltaV2Prediction) -> int:
    votes = 0
    if pred.boundary_score == 3:
        votes += 1
    if pred.classification <= 3:
        votes += 1
    if pred.delta_score < 0:
        votes += 1
    return votes


def reconstruct_history_prior_delta_v3_score(pred: HistoryPriorDeltaV2Prediction) -> int:
    """
    Hybrid reconstruction for v3.

    Keep the prior as the exact-score anchor, but expose DSAT discovery when
    multiple independent signals agree. Upward movement remains conservative.
    """
    score = clip_score(pred.history_prior_score)
    dsat_votes = history_prior_delta_v3_dsat_votes(pred)

    if dsat_votes >= 2:
        score = min(score, 3)
    elif pred.delta_confidence == "high":
        score += sign(pred.delta_score)
    elif pred.delta_confidence == "medium" and abs(pred.delta_score) == 2:
        score += sign(pred.delta_score)

    score = clip_score(score)
    if pred.boundary_confidence == "high" and pred.boundary_score == 4:
        score = max(score, 4)
    return clip_score(score)


def reconstruct_history_prior_delta_v3_1_score(pred: HistoryPriorDeltaV2Prediction) -> int:
    """
    Stricter v3 reconstruction.

    The v3 subset run showed that two-vote DSAT cases were sparse and noisy.
    V3.1 only forces <=3 when all three DSAT signals agree.
    """
    score = clip_score(pred.history_prior_score)
    dsat_votes = history_prior_delta_v3_dsat_votes(pred)

    if dsat_votes >= 3:
        score = min(score, 3)
    elif pred.delta_confidence == "high":
        score += sign(pred.delta_score)
    elif pred.delta_confidence == "medium" and abs(pred.delta_score) == 2:
        score += sign(pred.delta_score)

    score = clip_score(score)
    if pred.boundary_confidence == "high" and pred.boundary_score == 4:
        score = max(score, 4)
    return clip_score(score)


def retrieve_anchor_turns(
    retriever: AnchorRetriever | None,
    query_user_msg: str,
    query_assistant_reply: str,
    k: int,
    turn_eval_prompt_version: str,
) -> list[AnchorTurn] | None:
    if retriever is None or k <= 0:
        return None
    if turn_eval_prompt_version in {
        "history_prior_delta_v3_episodic",
        "history_prior_delta_v3_episodic_twopass",
    }:
        return retriever.retrieve_boundary_paired(
            query_user_msg=query_user_msg,
            query_assistant_reply=query_assistant_reply,
            k=k,
        )
    return retriever.retrieve(
        query_user_msg=query_user_msg,
        query_assistant_reply=query_assistant_reply,
        k=k,
    )


def anchor_metadata(anchors: list[AnchorTurn] | None) -> dict:
    if not anchors:
        return {
            "n_anchors_retrieved": 0,
            "anchor_scores": [],
            "anchor_tasks": [],
            "anchor_evidence_roles": [],
        }
    return {
        "n_anchors_retrieved": len(anchors),
        "anchor_scores": [a.score for a in anchors],
        "anchor_tasks": [a.task for a in anchors],
        "anchor_evidence_roles": [getattr(a, "evidence_role", "") for a in anchors],
    }


# Backward-compatible aliases for collect_personalized and older imports/tests.
_normalize_pred_reason = normalize_pred_reason
_clip_score = clip_score
_reconstruct_history_prior_delta_score = reconstruct_history_prior_delta_score
_sign = sign
_reconstruct_history_prior_delta_v2_score = reconstruct_history_prior_delta_v2_score
_history_prior_delta_v3_dsat_votes = history_prior_delta_v3_dsat_votes
_reconstruct_history_prior_delta_v3_score = reconstruct_history_prior_delta_v3_score
_reconstruct_history_prior_delta_v3_1_score = reconstruct_history_prior_delta_v3_1_score
_retrieve_anchor_turns = retrieve_anchor_turns
_anchor_metadata = anchor_metadata
