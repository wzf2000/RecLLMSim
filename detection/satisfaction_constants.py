REASON_TO_ID: dict[str, int] = {
    "其它": 0,
    "不够多样": 1,
    "不可用": 2,
    "满意": 3,
    "不够细致": 4,
    "不满足需求": 5,
}


def get_reason_to_id() -> dict[str, int]:
    # 返回副本，避免调用方意外修改全局常量
    return dict(REASON_TO_ID)


def get_id_to_reason() -> dict[int, str]:
    return {v: k for k, v in REASON_TO_ID.items()}
