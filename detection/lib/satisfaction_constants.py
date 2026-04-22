REASON_TO_ID: dict[str, int] = {
    "其它": 0,
    "不够多样": 1,
    "不可用": 2,
    "满意": 3,
    "不够细致": 4,
    "不满足需求": 5,
}

SATISFIED_REASON = "满意"


def get_reason_to_id() -> dict[str, int]:
    # 返回副本，避免调用方意外修改全局常量
    return dict(REASON_TO_ID)


def get_id_to_reason() -> dict[int, str]:
    return {v: k for k, v in REASON_TO_ID.items()}


def get_dissatisfied_reasons() -> list[str]:
    return [reason for reason in REASON_TO_ID if reason != SATISFIED_REASON]


def is_reason_valid_for_score(score: int, reason: str) -> bool:
    normalized = reason.strip()
    if score >= 4:
        return normalized == SATISFIED_REASON
    return normalized in REASON_TO_ID and normalized != SATISFIED_REASON


def normalize_reason_for_score(
    score: int,
    reason: str,
    default_reason: str = "其它",
) -> str:
    if score >= 4:
        return SATISFIED_REASON

    normalized = reason.strip()
    if normalized in REASON_TO_ID and normalized != SATISFIED_REASON:
        return normalized

    dissatisfied_reasons = get_dissatisfied_reasons()
    if default_reason in dissatisfied_reasons:
        return default_reason
    if dissatisfied_reasons:
        return dissatisfied_reasons[0]
    return SATISFIED_REASON
