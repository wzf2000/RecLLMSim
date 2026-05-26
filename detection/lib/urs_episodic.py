"""Raw episodic-memory retrieval for URS session-level satisfaction prediction."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from typing import Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .memory import _truncate
from .personalized_data import SessionData
from .urs_memory import _format_session_dialogue

UrsRetrievalStrategy = Literal[
    "topk_similar",
    "score_balanced",
    "boundary_paired",
    "nearest",
]


@dataclass(frozen=True)
class UrsEpisodicMemoryRecord:
    """One historical URS session kept as a retrievable memory item."""

    memory_id: str
    user: str
    source_task: str
    source_file: str
    task_context: str
    dialogue: str
    score: int
    reason: str
    score_side: str
    evidence_role: str = "similar"
    similarity: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def score_side(score: int) -> str:
    if score <= 3:
        return "dsat"
    if score == 4:
        return "sat_4"
    return "sat_5"


class UrsEpisodicMemoryIndex:
    """
    TF-IDF retrieval index over raw historical URS sessions for one user.

    This is intentionally not a summary memory. Every labeled historical
    session remains an independent evidence item for current-session scoring.
    """

    def __init__(
        self,
        user: str,
        sessions: list[SessionData],
        max_dialogue_chars: int = 900,
    ) -> None:
        self.user = user
        self.max_dialogue_chars = max_dialogue_chars
        self.records = self._build_records(sessions)
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        if self.records:
            self._vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                min_df=1,
                max_features=50000,
            )
            self._matrix = self._vectorizer.fit_transform(
                [self._record_to_text(r) for r in self.records]
            )

    def _build_records(self, sessions: list[SessionData]) -> list[UrsEpisodicMemoryRecord]:
        records: list[UrsEpisodicMemoryRecord] = []
        for idx, session in enumerate(sessions):
            if not session.satisfaction_scores:
                continue
            score = int(session.satisfaction_scores[0])
            reason = session.dissatisfaction_reasons[0] if session.dissatisfaction_reasons else "满意"
            source_file = os.path.basename(session.file_path)
            memory_id = f"{session.user}__{session.task}__{source_file}__session_{idx}"
            dialogue = _format_session_dialogue(
                session.history,
                max_chars=max(120, self.max_dialogue_chars // max(1, len(session.history))),
            )
            records.append(
                UrsEpisodicMemoryRecord(
                    memory_id=memory_id,
                    user=session.user,
                    source_task=session.task,
                    source_file=source_file,
                    task_context=session.task_context,
                    dialogue=_truncate(dialogue, self.max_dialogue_chars),
                    score=score,
                    reason=reason,
                    score_side=score_side(score),
                )
            )
        return records

    def _record_to_text(self, record: UrsEpisodicMemoryRecord) -> str:
        return "\n".join(
            [
                f"intent: {record.source_task}",
                f"task context: {record.task_context}",
                f"dialogue: {record.dialogue}",
            ]
        )

    def _query_to_text(self, session: SessionData) -> str:
        dialogue = _format_session_dialogue(session.history, max_chars=260)
        return "\n".join(
            [
                f"intent: {session.task}",
                f"task context: {session.task_context}",
                f"dialogue: {dialogue}",
            ]
        )

    def _ranked_records(self, session: SessionData) -> list[UrsEpisodicMemoryRecord]:
        if not self.records or self._vectorizer is None or self._matrix is None:
            return []
        query_vec = self._vectorizer.transform([self._query_to_text(session)])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        order = np.argsort(scores)[::-1]
        return [replace(self.records[i], similarity=float(scores[i])) for i in order]

    def retrieve(
        self,
        session: SessionData,
        k: int = 4,
        strategy: UrsRetrievalStrategy = "boundary_paired",
    ) -> list[UrsEpisodicMemoryRecord]:
        ranked = self._ranked_records(session)
        if k <= 0 or not ranked:
            return []
        if strategy in {"topk_similar", "nearest"}:
            return [replace(r, evidence_role="similar") for r in ranked[:k]]
        if strategy == "boundary_paired":
            return self._pick_boundary_paired(ranked, k)
        if strategy == "score_balanced":
            return self._pick_score_balanced(ranked, k)
        raise ValueError(f"Unknown URS retrieval strategy: {strategy}")

    def _pick_boundary_paired(
        self,
        ranked: list[UrsEpisodicMemoryRecord],
        k: int,
    ) -> list[UrsEpisodicMemoryRecord]:
        dsat_budget = max(1, k // 2)
        sat_budget = k - dsat_budget
        selected: list[UrsEpisodicMemoryRecord] = []
        selected.extend(
            replace(r, evidence_role="boundary_dsat")
            for r in ranked
            if r.score <= 3
        )
        selected = selected[:dsat_budget]
        selected.extend(
            replace(r, evidence_role="boundary_sat")
            for r in ranked
            if r.score >= 4
        )
        selected = selected[: dsat_budget + sat_budget]
        return self._fill_remaining(selected, ranked, k)

    def _pick_score_balanced(
        self,
        ranked: list[UrsEpisodicMemoryRecord],
        k: int,
    ) -> list[UrsEpisodicMemoryRecord]:
        buckets = [
            ("low_score", lambda r: r.score <= 3),
            ("score_4", lambda r: r.score == 4),
            ("score_5", lambda r: r.score == 5),
        ]
        selected: list[UrsEpisodicMemoryRecord] = []
        per_bucket = max(1, k // len(buckets))
        for role, pred in buckets:
            bucket = [replace(r, evidence_role=role) for r in ranked if pred(r)]
            selected.extend(bucket[:per_bucket])
        return self._fill_remaining(selected[:k], ranked, k)

    def _fill_remaining(
        self,
        selected: list[UrsEpisodicMemoryRecord],
        ranked: list[UrsEpisodicMemoryRecord],
        k: int,
    ) -> list[UrsEpisodicMemoryRecord]:
        seen = {r.memory_id for r in selected}
        result = list(selected)
        for r in ranked:
            if len(result) >= k:
                break
            if r.memory_id in seen:
                continue
            result.append(replace(r, evidence_role="fallback_similar"))
            seen.add(r.memory_id)
        return result


def format_urs_episodic_memories(
    retrieved: list[UrsEpisodicMemoryRecord],
    max_context_chars: int = 180,
    max_dialogue_chars: int = 520,
) -> str:
    if not retrieved:
        return "（没有可用的历史 episodic memory 证据。）"
    lines: list[str] = []
    for i, record in enumerate(retrieved, 1):
        lines.append(
            f"[E{i}] id={record.memory_id} role={record.evidence_role} "
            f"sim={record.similarity:.3f}"
        )
        lines.append(f"历史 intent：{record.source_task}")
        lines.append(f"历史任务背景：{_truncate(record.task_context, max_context_chars)}")
        lines.append(f"历史完整对话：\n{_truncate(record.dialogue, max_dialogue_chars)}")
        lines.append(f"真实满意度：{record.score}")
        lines.append(f"真实原因：{record.reason}")
        lines.append("")
    return "\n".join(lines).strip()
