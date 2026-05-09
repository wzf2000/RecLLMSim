from __future__ import annotations

from pydantic import BaseModel, Field


class TraceAnswer(BaseModel):
    classification: int = Field(ge=1, le=5)
    reason: str
    analysis: str


class ReflectionAnswer(BaseModel):
    problem_analysis: str
    reflection: str
    revised_reasoning: str


