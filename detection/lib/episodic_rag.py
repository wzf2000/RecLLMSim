"""Raw episodic-memory retrieval for personalized satisfaction prediction."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from typing import Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .memory_formatting import _truncate
from .personalized_data import SessionData

RetrievalStrategy = Literal[
    "topk_similar",
    "score_balanced",
    "boundary_paired",
    "nearest",
]


@dataclass(frozen=True)
class EpisodicMemoryRecord:
    """One historical assistant turn used as a retrievable memory item."""

    memory_id: str
    user: str
    source_task: str
    source_file: str
    turn_idx: int
    task_context: str
    local_history: list[dict]
    last_user_msg: str
    assistant_reply: str
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


class EpisodicMemoryIndex:
    """
    TF-IDF retrieval index over raw historical assistant turns for one user.

    Unlike UserMemory, this index does not summarize or update memory. Each
    historical labeled assistant turn remains an independent retrievable item.
    """

    def __init__(
        self,
        user: str,
        sessions: list[SessionData],
        local_history_size: int = 4,
    ) -> None:
        self.user = user
        self.local_history_size = local_history_size
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

    def _build_records(self, sessions: list[SessionData]) -> list[EpisodicMemoryRecord]:
        records: list[EpisodicMemoryRecord] = []
        for session in sessions:
            assistant_idx = 0
            last_user_msg = ""
            seen_history: list[dict] = []
            for utt in session.history:
                role = utt.get("role", "")
                content = utt.get("content", "")
                if role == "user":
                    last_user_msg = content
                    seen_history.append({"role": role, "content": content})
                    continue
                if role != "assistant":
                    seen_history.append({"role": role, "content": content})
                    continue
                if assistant_idx >= len(session.satisfaction_scores):
                    seen_history.append({"role": role, "content": content})
                    continue

                score = int(session.satisfaction_scores[assistant_idx])
                reason = session.dissatisfaction_reasons[assistant_idx]
                source_file = os.path.basename(session.file_path)
                memory_id = (
                    f"{session.user}__{session.task}__{source_file}"
                    f"__turn_{assistant_idx}"
                )
                local_history = seen_history[-self.local_history_size:]
                records.append(
                    EpisodicMemoryRecord(
                        memory_id=memory_id,
                        user=session.user,
                        source_task=session.task,
                        source_file=source_file,
                        turn_idx=assistant_idx,
                        task_context=session.task_context,
                        local_history=local_history,
                        last_user_msg=last_user_msg,
                        assistant_reply=content,
                        score=score,
                        reason=reason,
                        score_side=score_side(score),
                    )
                )
                seen_history.append({"role": role, "content": content})
                assistant_idx += 1
        return records

    def _record_to_text(self, record: EpisodicMemoryRecord) -> str:
        return "\n".join(
            [
                f"任务背景：{record.task_context}",
                f"用户请求：{record.last_user_msg}",
                f"用户请求：{record.last_user_msg}",
                f"助手回复：{record.assistant_reply}",
            ]
        )

    def _query_to_text(
        self,
        task_context: str,
        last_user_msg: str,
        assistant_reply: str,
    ) -> str:
        return "\n".join(
            [
                f"任务背景：{task_context}",
                f"用户请求：{last_user_msg}",
                f"用户请求：{last_user_msg}",
                f"助手回复：{assistant_reply}",
            ]
        )

    def _ranked_records(
        self,
        task_context: str,
        last_user_msg: str,
        assistant_reply: str,
    ) -> list[EpisodicMemoryRecord]:
        if not self.records or self._vectorizer is None or self._matrix is None:
            return []
        query_vec = self._vectorizer.transform(
            [self._query_to_text(task_context, last_user_msg, assistant_reply)]
        )
        scores = cosine_similarity(query_vec, self._matrix)[0]
        order = np.argsort(scores)[::-1]
        return [
            replace(self.records[i], similarity=float(scores[i]))
            for i in order
        ]

    def retrieve(
        self,
        task_context: str,
        last_user_msg: str,
        assistant_reply: str,
        k: int = 6,
        strategy: RetrievalStrategy = "topk_similar",
    ) -> list[EpisodicMemoryRecord]:
        ranked = self._ranked_records(task_context, last_user_msg, assistant_reply)
        if k <= 0 or not ranked:
            return []
        if strategy in {"topk_similar", "nearest"}:
            return [
                replace(r, evidence_role="similar")
                for r in ranked[:k]
            ]
        if strategy == "boundary_paired":
            return self._pick_boundary_paired(ranked, k)
        if strategy == "score_balanced":
            return self._pick_score_balanced(ranked, k)
        raise ValueError(f"Unknown retrieval strategy: {strategy}")

    def _pick_boundary_paired(
        self,
        ranked: list[EpisodicMemoryRecord],
        k: int,
    ) -> list[EpisodicMemoryRecord]:
        dsat_budget = max(1, k // 2)
        sat_budget = k - dsat_budget
        selected: list[EpisodicMemoryRecord] = []
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
        selected = selected[:k]
        return self._fill_remaining(selected, ranked, k)

    def _pick_score_balanced(
        self,
        ranked: list[EpisodicMemoryRecord],
        k: int,
    ) -> list[EpisodicMemoryRecord]:
        buckets = [
            ("low_score", lambda r: r.score <= 3),
            ("score_4", lambda r: r.score == 4),
            ("score_5", lambda r: r.score == 5),
        ]
        selected: list[EpisodicMemoryRecord] = []
        per_bucket = max(1, k // len(buckets))
        for role, pred in buckets:
            bucket = [replace(r, evidence_role=role) for r in ranked if pred(r)]
            selected.extend(bucket[:per_bucket])
        return self._fill_remaining(selected[:k], ranked, k)

    def _fill_remaining(
        self,
        selected: list[EpisodicMemoryRecord],
        ranked: list[EpisodicMemoryRecord],
        k: int,
    ) -> list[EpisodicMemoryRecord]:
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


def predict_score_from_retrieval(
    retrieved: list[EpisodicMemoryRecord],
    default_score: int = 4,
) -> int:
    """Nearest-neighbor score baseline for fast no-LLM smoke/evaluation."""
    if not retrieved:
        return default_score
    weights = np.array([max(r.similarity, 0.0) + 1e-3 for r in retrieved])
    scores = np.array([r.score for r in retrieved], dtype=float)
    pred = int(round(float(np.dot(weights, scores) / weights.sum())))
    return min(5, max(1, pred))


def format_episodic_memories(
    retrieved: list[EpisodicMemoryRecord],
    max_user_chars: int = 180,
    max_reply_chars: int = 260,
) -> str:
    if not retrieved:
        return "（没有可用的历史记忆证据。）"
    lines: list[str] = []
    for i, record in enumerate(retrieved, 1):
        lines.append(
            f"[E{i}] id={record.memory_id} role={record.evidence_role} "
            f"sim={record.similarity:.3f}"
        )
        lines.append(f"任务：{record.source_task}")
        lines.append(f"历史用户请求：{_truncate(record.last_user_msg, max_user_chars)}")
        lines.append(f"历史助手回复：{_truncate(record.assistant_reply, max_reply_chars)}")
        lines.append(f"真实满意度：{record.score}")
        lines.append(f"真实原因：{record.reason}")
        lines.append("")
    return "\n".join(lines).strip()
