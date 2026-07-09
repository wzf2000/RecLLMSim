"""
个性化满意度感知任务数据模块

数据划分策略：Cross-Task Split
  - 对每个 (用户, 目标任务类型) 对：
      * history_sessions = 该用户在其他任务类型中的所有 session（满意度标签可见）
      * target_sessions  = 该用户在目标任务类型中的所有 session（满意度标签待预测）
  - 用户层面：80% test / 20% train（sklearn GroupShuffleSplit）

数据结构：
  SessionData       — 单个 session 的全量信息
  PersonalizedSample — 一个 (用户, 目标任务) 对：包含 target_sessions + history_sessions
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace

from loguru import logger
from sklearn.model_selection import GroupShuffleSplit

from .utils import HUMAN_DIR

TASK_LIST: list[str] = ["旅行规划", "礼物准备", "菜谱规划", "技能学习规划"]


# ──────────────────────────────────────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SessionData:
    """单个对话 session 的全量信息。"""
    user: str
    task: str
    file_path: str
    task_context: str
    profile: dict
    history: list[dict]           # [{"role": ..., "content": ...}]
    satisfaction_scores: list[int]
    dissatisfaction_reasons: list[str]
    chat_model: str

    @property
    def assistant_turns(self) -> int:
        return len(self.satisfaction_scores)

    @property
    def avg_satisfaction(self) -> float:
        if not self.satisfaction_scores:
            return 0.0
        return sum(self.satisfaction_scores) / len(self.satisfaction_scores)


@dataclass
class PersonalizedSample:
    """
    个性化预测的基本单元：一个 (用户, 目标任务类型) 对。

    history_sessions: 来自其他任务类型的所有 session，满意度标签可见
    target_sessions:  目标任务类型的所有 session，满意度标签待预测
    """
    user: str
    profile: dict
    target_task: str
    history_sessions: list[SessionData]
    target_sessions: list[SessionData]
    history_session_budget: int = 0
    history_budget_strategy: str = "all"
    n_history_sessions_before_budget: int | None = None

    # 唯一标识，用于跨运行去重
    @property
    def block_id(self) -> str:
        return f"{self.user}__{self.target_task}"

    @property
    def n_history_sessions(self) -> int:
        return len(self.history_sessions)

    @property
    def n_target_turns(self) -> int:
        return sum(s.assistant_turns for s in self.target_sessions)

    @property
    def history_tasks(self) -> list[str]:
        return list(dict.fromkeys(s.task for s in self.history_sessions))

    @property
    def history_cache_tag(self) -> str:
        if self.history_session_budget <= 0:
            return ""
        strategy = self.history_budget_strategy.replace("/", "_").replace(" ", "_")
        return f"histk{self.history_session_budget}_{strategy}"


# ──────────────────────────────────────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────────────────────────────────────

def load_all_sessions(
    data_dir: str = HUMAN_DIR,
) -> dict[str, dict[str, list[SessionData]]]:
    """
    加载所有 session。

    返回 {user: {task: [SessionData, ...]}}，每个 user/task 下的 session
    按文件名（数字序）排序。
    """
    result: dict[str, dict[str, list[SessionData]]] = {}

    users = sorted(os.listdir(data_dir))
    for user in users:
        user_dir = os.path.join(data_dir, user)
        if not os.path.isdir(user_dir):
            continue
        result[user] = {}
        for task in TASK_LIST:
            task_dir = os.path.join(user_dir, task)
            if not os.path.exists(task_dir):
                result[user][task] = []
                continue
            files = sorted(
                [f for f in os.listdir(task_dir) if f.endswith(".json")],
                key=lambda x: int(x.split(".")[0]),
            )
            sessions: list[SessionData] = []
            for fname in files:
                fpath = os.path.join(task_dir, fname)
                with open(fpath, "r", encoding="utf-8") as fp:
                    data = json.load(fp)

                # 跳过没有 questionnaire（未标注）的文件
                if "questionnaire" not in data:
                    continue

                history = [
                    {"role": utt["role"], "content": utt["content"]}
                    for utt in data["history"]
                ]
                satisfaction_scores: list[int] = []
                dissatisfaction_reasons: list[str] = []
                for utt in data["history"]:
                    if utt["role"] != "assistant":
                        continue
                    score = int(utt["satisfaction"])
                    satisfaction_scores.append(score)
                    if score <= 3:
                        dissatisfaction_reasons.append(utt.get("reason", "其它"))
                    else:
                        dissatisfaction_reasons.append("满意")

                sessions.append(
                    SessionData(
                        user=user,
                        task=task,
                        file_path=fpath,
                        task_context=data.get("task_context", ""),
                        profile=data.get("profile", {}),
                        history=history,
                        satisfaction_scores=satisfaction_scores,
                        dissatisfaction_reasons=dissatisfaction_reasons,
                        chat_model=data.get("chat_model", "unknown"),
                    )
                )
            result[user][task] = sessions

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Cross-Task Split + 用户级 train/test 划分
# ──────────────────────────────────────────────────────────────────────────────

def build_personalized_samples(
    split: str = "test",
    train_ratio: float = 0.2,
    seed: int = 42,
    min_history_sessions: int = 1,
    target_tasks: list[str] | None = None,
    data_dir: str = HUMAN_DIR,
) -> list[PersonalizedSample]:
    """
    构建个性化预测样本列表。

    划分策略（Cross-Task）：
      - history_sessions = 该用户在其他 3 个任务类型中的所有 session
      - target_sessions  = 该用户在目标任务类型中的所有 session

    参数
    ----
    split : "train" | "test" | "all"
        数据集划分；用户层面 GroupShuffleSplit，不重叠。
    train_ratio : float
        训练集用户比例，默认 0.2（即 80% 用于 test）。
    seed : int
        随机种子。
    min_history_sessions : int
        过滤：历史 session 数量必须 >= 该值，否则跳过该 (用户, 目标任务) 对。
    target_tasks : list[str] | None
        限定只生成哪些任务作为目标任务；None 表示全部 4 个任务。
    """
    if target_tasks is None:
        target_tasks = TASK_LIST

    all_sessions = load_all_sessions(data_dir)

    # 第一步：构建所有候选 PersonalizedSample（所有用户）
    all_samples: list[PersonalizedSample] = []
    for user, task_sessions in all_sessions.items():
        profile = None
        for task in TASK_LIST:
            if task_sessions.get(task):
                profile = task_sessions[task][0].profile
                break
        if profile is None:
            continue

        for target_task in target_tasks:
            target_sessions = task_sessions.get(target_task, [])
            if not target_sessions:
                continue

            history_sessions: list[SessionData] = []
            for other_task in TASK_LIST:
                if other_task == target_task:
                    continue
                history_sessions.extend(task_sessions.get(other_task, []))

            if len(history_sessions) < min_history_sessions:
                continue

            all_samples.append(
                PersonalizedSample(
                    user=user,
                    profile=profile,
                    target_task=target_task,
                    history_sessions=history_sessions,
                    target_sessions=target_sessions,
                )
            )

    if not all_samples:
        return []

    if split == "all":
        logger.info(
            f"[personalized_data] split=all, samples={len(all_samples)}, "
            f"users={len({s.user for s in all_samples})}"
        )
        return all_samples

    # 第二步：用户级 GroupShuffleSplit（按唯一用户划分，不按样本）
    user_list = [s.user for s in all_samples]
    unique_users = sorted(set(user_list))

    # 为每个样本建立用户 index（GroupShuffleSplit 以 groups 参数分组）
    gss = GroupShuffleSplit(
        n_splits=1,
        train_size=train_ratio,
        test_size=1.0 - train_ratio,
        random_state=seed,
    )
    indices = list(range(len(all_samples)))
    train_rel, test_rel = next(gss.split(indices, groups=user_list))

    split_map = {
        "train": [all_samples[i] for i in train_rel],
        "test":  [all_samples[i] for i in test_rel],
    }
    selected = split_map[split]

    train_users = {all_samples[i].user for i in train_rel}
    test_users  = {all_samples[i].user for i in test_rel}
    logger.info(
        f"[personalized_data] split={split}, "
        f"train_users={len(train_users)}, test_users={len(test_users)}, "
        f"selected_samples={len(selected)}, "
        f"total_target_turns={sum(s.n_target_turns for s in selected)}"
    )
    return selected


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def _select_history_sessions_round_robin_by_task(
    history_sessions: list[SessionData],
    budget: int,
) -> list[SessionData]:
    sessions_by_task: dict[str, list[SessionData]] = {}
    for session in history_sessions:
        sessions_by_task.setdefault(session.task, []).append(session)

    task_order = [task for task in TASK_LIST if task in sessions_by_task]
    selected: list[SessionData] = []
    depth = 0
    while len(selected) < budget:
        added = False
        for task in task_order:
            task_sessions = sessions_by_task[task]
            if depth < len(task_sessions):
                selected.append(task_sessions[depth])
                added = True
                if len(selected) >= budget:
                    break
        if not added:
            break
        depth += 1
    return selected


def apply_history_session_budget(
    samples: list[PersonalizedSample],
    history_session_budget: int,
    strategy: str = "round_robin_task",
) -> list[PersonalizedSample]:
    """
    Return samples with at most K source-history sessions per user-target block.

    This is used for robustness ablations under sparse user histories. The default
    round-robin strategy keeps the selection deterministic while spreading small
    budgets across source task types when possible.
    """
    if history_session_budget <= 0:
        return samples
    if strategy not in {"round_robin_task", "original_order"}:
        raise ValueError(f"Unknown history budget strategy: {strategy}")

    budgeted_samples: list[PersonalizedSample] = []
    for sample in samples:
        original_count = len(sample.history_sessions)
        if strategy == "round_robin_task":
            selected_history = _select_history_sessions_round_robin_by_task(
                sample.history_sessions,
                history_session_budget,
            )
        else:
            selected_history = sample.history_sessions[:history_session_budget]

        budgeted_samples.append(
            replace(
                sample,
                history_sessions=selected_history,
                history_session_budget=history_session_budget,
                history_budget_strategy=strategy,
                n_history_sessions_before_budget=original_count,
            )
        )

    if not budgeted_samples:
        return []

    before_counts = [
        s.n_history_sessions_before_budget or s.n_history_sessions
        for s in budgeted_samples
    ]
    after_counts = [s.n_history_sessions for s in budgeted_samples]
    logger.info(
        f"[personalized_data] applied history_session_budget={history_session_budget}, "
        f"strategy={strategy}, avg_history_sessions "
        f"{sum(before_counts) / len(before_counts):.2f}->{sum(after_counts) / len(after_counts):.2f}"
    )
    return budgeted_samples

def get_all_users(data_dir: str = HUMAN_DIR) -> list[str]:
    return sorted(
        u for u in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, u))
    )


def dataset_stats(samples: list[PersonalizedSample]) -> dict:
    """返回数据集统计信息（用于日志/报告）。"""
    users = {s.user for s in samples}
    target_sessions = sum(len(s.target_sessions) for s in samples)
    target_turns = sum(s.n_target_turns for s in samples)
    history_per_sample = [s.n_history_sessions for s in samples]
    stats = {
        "n_users": len(users),
        "n_blocks": len(samples),
        "n_target_sessions": target_sessions,
        "n_target_turns": target_turns,
        "avg_history_sessions": sum(history_per_sample) / len(history_per_sample) if history_per_sample else 0,
        "min_history_sessions": min(history_per_sample) if history_per_sample else 0,
        "max_history_sessions": max(history_per_sample) if history_per_sample else 0,
    }
    original_history_per_sample = [
        s.n_history_sessions_before_budget
        for s in samples
        if s.n_history_sessions_before_budget is not None
    ]
    if original_history_per_sample:
        stats.update(
            {
                "avg_history_sessions_before_budget": (
                    sum(original_history_per_sample) / len(original_history_per_sample)
                ),
                "min_history_sessions_before_budget": min(original_history_per_sample),
                "max_history_sessions_before_budget": max(original_history_per_sample),
            }
        )
    return stats
