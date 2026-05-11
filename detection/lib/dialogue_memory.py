"""Unlabeled dialogue-memory retrieval for static replay generation."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from typing import Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .memory_formatting import _truncate
from .personalized_data import SessionData

DialogueMemoryStrategy = Literal["tfidf", "diverse"]


@dataclass(frozen=True)
class DialogueMemoryRecord:
    """One unlabeled historical dialogue episode from another scenario."""

    memory_id: str
    user: str
    source_task: str
    source_file: str
    turn_idx: int
    local_history: list[dict]
    last_user_msg: str
    assistant_reply: str
    evidence_role: str = "similar"
    similarity: float = 0.0

    def to_metadata(self, max_chars: int = 240) -> dict:
        data = asdict(self)
        data["local_history"] = [
            {
                "role": m.get("role", ""),
                "content": _truncate(str(m.get("content", "")), max_chars),
            }
            for m in self.local_history
        ]
        data["last_user_msg"] = _truncate(self.last_user_msg, max_chars)
        data["assistant_reply"] = _truncate(self.assistant_reply, max_chars)
        return data


class DialogueMemoryIndex:
    """
    TF-IDF index over raw historical dialogue turns for one user.

    This index intentionally excludes satisfaction scores, dissatisfaction
    reasons, and user profiles. It is meant for static replay candidate models,
    where only previous conversations should be visible as memory.
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
                [self._record_to_text(record) for record in self.records]
            )

    def _build_records(self, sessions: list[SessionData]) -> list[DialogueMemoryRecord]:
        records: list[DialogueMemoryRecord] = []
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
                source_file = os.path.basename(session.file_path)
                memory_id = (
                    f"{session.user}__{session.task}__{source_file}"
                    f"__turn_{assistant_idx}"
                )
                records.append(
                    DialogueMemoryRecord(
                        memory_id=memory_id,
                        user=session.user,
                        source_task=session.task,
                        source_file=source_file,
                        turn_idx=assistant_idx,
                        local_history=seen_history[-self.local_history_size:],
                        last_user_msg=last_user_msg,
                        assistant_reply=content,
                    )
                )
                seen_history.append({"role": role, "content": content})
                assistant_idx += 1
        return records

    def _record_to_text(self, record: DialogueMemoryRecord) -> str:
        local_text = "\n".join(
            f"{m.get('role', '')}: {m.get('content', '')}"
            for m in record.local_history
        )
        return "\n".join(
            [
                f"来源任务：{record.source_task}",
                local_text,
                f"user: {record.last_user_msg}",
                f"assistant: {record.assistant_reply}",
            ]
        )

    def _query_to_text(self, dialogue_prefix: list[dict]) -> str:
        prefix_tail = dialogue_prefix[-8:]
        local_text = "\n".join(
            f"{m.get('role', '')}: {m.get('content', '')}"
            for m in prefix_tail
        )
        last_user_msg = ""
        for message in reversed(dialogue_prefix):
            if message.get("role") == "user":
                last_user_msg = str(message.get("content", ""))
                break
        return "\n".join([local_text, f"user: {last_user_msg}", f"user: {last_user_msg}"])

    def _ranked_records(self, dialogue_prefix: list[dict]) -> list[DialogueMemoryRecord]:
        if not self.records or self._vectorizer is None or self._matrix is None:
            return []
        query_vec = self._vectorizer.transform([self._query_to_text(dialogue_prefix)])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        order = np.argsort(scores)[::-1]
        return [
            replace(self.records[i], similarity=float(scores[i]))
            for i in order
        ]

    def retrieve(
        self,
        dialogue_prefix: list[dict],
        k: int = 4,
        strategy: DialogueMemoryStrategy = "tfidf",
    ) -> list[DialogueMemoryRecord]:
        ranked = self._ranked_records(dialogue_prefix)
        if k <= 0 or not ranked:
            return []
        if strategy == "tfidf":
            return [replace(record, evidence_role="similar") for record in ranked[:k]]
        if strategy == "diverse":
            return self._pick_diverse(ranked, k)
        raise ValueError(f"Unknown dialogue memory strategy: {strategy}")

    def _pick_diverse(
        self,
        ranked: list[DialogueMemoryRecord],
        k: int,
    ) -> list[DialogueMemoryRecord]:
        by_task: dict[str, list[DialogueMemoryRecord]] = {}
        for record in ranked:
            by_task.setdefault(record.source_task, []).append(record)

        selected: list[DialogueMemoryRecord] = []
        seen: set[str] = set()
        task_order = sorted(
            by_task,
            key=lambda task: by_task[task][0].similarity,
            reverse=True,
        )
        while len(selected) < k:
            added = False
            for task in task_order:
                bucket = by_task[task]
                while bucket and bucket[0].memory_id in seen:
                    bucket.pop(0)
                if not bucket:
                    continue
                selected.append(replace(bucket.pop(0), evidence_role="diverse_task"))
                seen.add(selected[-1].memory_id)
                added = True
                if len(selected) >= k:
                    break
            if not added:
                break

        for record in ranked:
            if len(selected) >= k:
                break
            if record.memory_id in seen:
                continue
            selected.append(replace(record, evidence_role="fallback_similar"))
            seen.add(record.memory_id)
        return selected


def format_dialogue_memory_prompt(
    memories: list[DialogueMemoryRecord],
    max_chars_per_item: int = 700,
) -> str:
    if not memories:
        return ""

    blocks = [
        "You are continuing the current conversation by answering the user's latest message.",
        "Below are unlabeled past dialogue memories from the same user in other scenarios.",
        "Use them only to infer stable communication preferences and response style expectations.",
        "Do not mention these memories, do not copy irrelevant details, and do not assume they are current facts.",
        "No satisfaction scores, reasons, or user profile annotations are provided.",
        "",
        "【Past dialogue memories】",
    ]
    for idx, memory in enumerate(memories, 1):
        history_lines = []
        for message in memory.local_history:
            role = str(message.get("role", ""))
            content = _truncate(str(message.get("content", "")), max_chars_per_item // 2)
            history_lines.append(f"{role}: {content}")
        assistant_reply = _truncate(memory.assistant_reply, max_chars_per_item)
        blocks.extend(
            [
                f"[{idx}] source_task={memory.source_task}, turn={memory.turn_idx}",
                "Context snippet:",
                *history_lines,
                f"assistant: {assistant_reply}",
                "",
            ]
        )
    return "\n".join(blocks).strip()
