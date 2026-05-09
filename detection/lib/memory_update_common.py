"""User-memory update prompt builders and update evidence helpers."""

from __future__ import annotations

import json
from collections import defaultdict

from .personalized_data import SessionData
from .satisfaction_constants import SATISFIED_REASON
from .memory_schema import UserMemory
from .memory_formatting import _truncate


def _collect_update_turns(
    new_session: SessionData,
    turn_predictions: list[dict],
    use_oracle_labels: bool,
) -> list[dict]:
    """抽取 update 所需的 turn 级证据，分数字段优先使用 gold（oracle）否则使用 pred。"""
    turns: list[dict] = []
    assistant_idx = 0
    last_user = ""
    for utt in new_session.history:
        if utt["role"] == "user":
            last_user = utt["content"]
        elif utt["role"] == "assistant" and assistant_idx < len(turn_predictions):
            pred = turn_predictions[assistant_idx]
            score = int(pred["gold_score"] if use_oracle_labels else pred["pred_score"])
            if use_oracle_labels:
                reason = pred.get("gold_reason", SATISFIED_REASON if score >= 4 else "其它")
            else:
                reason = pred.get("pred_reason", SATISFIED_REASON if score >= 4 else "其它")
            turns.append({
                "turn_idx": assistant_idx,
                "task": new_session.task,
                "user_msg": last_user,
                "assistant_reply": utt["content"],
                "score": score,
                "reason": reason,
                "pred_score": int(pred.get("pred_score", score)),
                "pred_reason": pred.get("pred_reason", reason),
                "analysis": pred.get("analysis", ""),
            })
            assistant_idx += 1
    return turns


def _format_update_examples(
    title: str,
    turns: list[dict],
    max_examples: int = 2,
) -> str:
    if not turns:
        return f"{title}\n  （无）"
    lines = [title]
    for i, t in enumerate(turns[:max_examples], 1):
        reason_suffix = f"（{t['reason']}）" if t["score"] <= 3 else ""
        lines.append(f"  [{i}] ★{t['score']}{reason_suffix}")
        lines.append(f"      用户：{_truncate(t['user_msg'], 120)}")
        lines.append(f"      助手：{_truncate(t['assistant_reply'])}")
    return "\n".join(lines)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


_GENERIC_REQUIREMENT_FRAGMENTS = (
    "详细", "具体", "清晰", "结构化", "实用", "全面", "完整", "分点", "有条理",
)


def _is_generic_requirement(text: str) -> bool:
    normalized = text.strip()
    if not normalized:
        return True
    if len(normalized) <= 4:
        return True
    return any(fragment in normalized for fragment in _GENERIC_REQUIREMENT_FRAGMENTS)


def _count_analysis_borderline_mentions(turn_predictions: list[dict]) -> int:
    keywords = ("基线", "边界", "接近", "未达到4", "未达到 4", "达到4", "达到 4", "未达到5", "未达到 5")
    count = 0
    for pred in turn_predictions:
        analysis = str(pred.get("analysis", ""))
        if any(k in analysis for k in keywords):
            count += 1
    return count


def _build_update_evidence_bundle(
    existing_memory: UserMemory,
    new_session: SessionData,
    turn_predictions: list[dict],
    use_oracle_labels: bool,
) -> dict:
    turns = _collect_update_turns(new_session, turn_predictions, use_oracle_labels)
    by_score: dict[int, list[dict]] = defaultdict(list)
    for t in turns:
        by_score[t["score"]].append(t)

    def _format_examples(examples: list[dict], limit: int = 2) -> list[dict]:
        out: list[dict] = []
        for t in examples[:limit]:
            out.append({
                "turn_idx": t["turn_idx"],
                "score": t["score"],
                "reason": t["reason"],
                "user_request": _truncate(t["user_msg"], 120),
                "assistant_reply_excerpt": _truncate(t["assistant_reply"], 160),
                "analysis_signal": _truncate(t.get("analysis", ""), 120),
            })
        return out

    score_sequence = [t["score"] for t in turns]
    score_flip_34 = sum(
        1 for a, b in zip(score_sequence, score_sequence[1:])
        if {a, b} == {3, 4}
    )
    score_flip_45 = sum(
        1 for a, b in zip(score_sequence, score_sequence[1:])
        if {a, b} == {4, 5}
    )
    pred_distribution = {
        f"score_{s}": sum(1 for t in turns if t["score"] == s)
        for s in range(1, 6)
    }
    dominant_score = max(pred_distribution.items(), key=lambda kv: kv[1])[0] if turns else "score_4"

    return {
        "session_summary": {
            "task": new_session.task,
            "n_turns": len(turns),
            "score_sequence": score_sequence,
            "mean_score": round(sum(score_sequence) / len(score_sequence), 4) if score_sequence else 0.0,
            "min_score": min(score_sequence) if score_sequence else None,
            "max_score": max(score_sequence) if score_sequence else None,
            "task_context_excerpt": _truncate(new_session.task_context, 160),
            "uses_oracle_labels": use_oracle_labels,
        },
        "current_memory_summary": {
            "avg_satisfaction_score": existing_memory.avg_satisfaction_score,
            "score_distribution": existing_memory.score_distribution.model_dump(),
            "scoring_style": existing_memory.scoring_style,
        },
        "boundary_evidence": {
            "n_leq3": len(by_score[1]) + len(by_score[2]) + len(by_score[3]),
            "n_4": len(by_score[4]),
            "n_5": len(by_score[5]),
            "has_3_and_4": bool(by_score[3] and by_score[4]),
            "has_4_and_5": bool(by_score[4] and by_score[5]),
            "low_examples": _format_examples(by_score[1] + by_score[2] + by_score[3]),
            "mid_examples": _format_examples(by_score[4]),
            "high_examples": _format_examples(by_score[5]),
        },
        "uncertainty_signals": {
            "predicted_score_only": not use_oracle_labels,
            "analysis_borderline_mentions": _count_analysis_borderline_mentions(turn_predictions),
            "score_flip_3_4": score_flip_34,
            "score_flip_4_5": score_flip_45,
            "dominant_score": dominant_score,
            "dominant_score_ratio": (
                round(max(pred_distribution.values()) / len(turns), 4) if turns else 0.0
            ),
        },
    }


