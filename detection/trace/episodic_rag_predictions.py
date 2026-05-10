"""Structured schemas for episodic-RAG turn prediction."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class EpisodicRagTurnPrediction(BaseModel):
    classification: int = Field(ge=1, le=5)
    reason: str
    analysis: str
    boundary_side: Literal["sat", "dsat"]
    evidence_confidence: Literal["low", "medium", "high"]

    @model_validator(mode="after")
    def check_boundary_consistency(self) -> "EpisodicRagTurnPrediction":
        expected = "sat" if self.classification >= 4 else "dsat"
        if self.boundary_side != expected:
            raise ValueError(
                "boundary_side must be sat when classification>=4 "
                "and dsat when classification<=3"
            )
        return self


class EpisodicRagBoundaryFirstPrediction(BaseModel):
    boundary_decision: Literal["sat", "dsat"]
    boundary_confidence: Literal["low", "medium", "high"]
    score_refinement: Literal[
        "severe_dsat",
        "clear_dsat",
        "near_boundary_dsat",
        "qualified_sat",
        "strong_sat",
    ]
    classification: int = Field(ge=1, le=5)
    reason: str
    analysis: str
    key_evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_boundary_consistency(self) -> "EpisodicRagBoundaryFirstPrediction":
        expected = "sat" if self.classification >= 4 else "dsat"
        if self.boundary_decision != expected:
            raise ValueError(
                "boundary_decision must be sat when classification>=4 "
                "and dsat when classification<=3"
            )
        refinement_to_scores = {
            "severe_dsat": {1},
            "clear_dsat": {2},
            "near_boundary_dsat": {3},
            "qualified_sat": {4},
            "strong_sat": {5},
        }
        if self.classification not in refinement_to_scores[self.score_refinement]:
            raise ValueError("score_refinement must match classification")
        return self
