from collections.abc import Sequence

from sklearn.model_selection import GroupShuffleSplit


def split_by_user_group_shuffle_split(
    users: Sequence[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """使用 sklearn 的 GroupShuffleSplit 按 user 分组切分到 train/val/test。

    约束：同一 user 不会跨集合。
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1")
    n = len(users)
    if n == 0:
        return [], [], []

    all_idx = list(range(n))
    gss_1 = GroupShuffleSplit(
        n_splits=1,
        train_size=train_ratio,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
    )
    train_rel, temp_rel = next(gss_1.split(all_idx, groups=users))
    train_idx = [all_idx[i] for i in train_rel]
    temp_idx = [all_idx[i] for i in temp_rel]

    temp_users = [users[i] for i in temp_idx]
    gss_2 = GroupShuffleSplit(
        n_splits=1,
        train_size=val_ratio / (val_ratio + test_ratio),
        random_state=seed,
    )
    val_rel, test_rel = next(gss_2.split(list(range(len(temp_idx))), groups=temp_users))
    valid_idx = [temp_idx[i] for i in val_rel]
    test_idx = [temp_idx[i] for i in test_rel]
    return train_idx, valid_idx, test_idx
