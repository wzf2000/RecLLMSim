"""
USS (User Satisfaction Simulation) 数据加载模块。

提供 get_uss_data()，返回与 get_satisfaction_data() 相同结构的 list[dict]，
可直接传入 preprocess_to_dict_data()（predictor/lora_ordinal.py, predictor/bert.py）。

USS 与项目原始数据的映射关系：
  profile        → 空占位 dict（USS 无用户画像）
  task_context   → 数据集名称 + 领域描述
  history        → 对话历史（role: user/assistant）
  satisfaction_scores     → 每个 assistant turn 的满意度（1-5 众数）
  dissatisfaction_reasons → "满意"(≥4) 或 "其它"(≤3)（USS 无细粒度原因）
  user           → dialogue_id（用于 GroupShuffleSplit 按对话切分）
"""

import json
import os
from pathlib import Path

from loguru import logger


# USS 无用户画像，使用空占位 profile，与 format_profile() 接口兼容
_DUMMY_PROFILE: dict = {
    "gender": "",
    "age": "",
    "background": "",
    "personality": [],
    "occupation": "",
    "daily_interests": [],
    "travel_habits": [],
    "dining_preferences": [],
    "spending_habits": [],
    "other_aspects": [],
}

_DEFAULT_USS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "uss", "processed"
)

ALL_DATASETS = ["JDDC", "SGD", "MWOZ", "ReDial", "CCPE"]


def _score_to_reason(score: int) -> str:
    """USS 无细粒度原因，按分数映射到项目已有 REASON_TO_ID 的两个标签。"""
    return "满意" if score >= 4 else "其它"


def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _group_by_dialogue(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in records:
        did = r["dialogue_id"]
        groups.setdefault(did, []).append(r)
    return groups


def _build_sample(dialogue_id: str, turn_records: list[dict]) -> dict | None:
    """
    将同一 dialogue 的 turn-level records 组装成 dialogue-level sample。

    turn_records 已按 turn_idx 排序，每条对应一个 assistant turn 被评估。
    history 通过各 record 的 history + assistant_reply 重建为完整对话序列，
    去重后得到线性历史。

    返回 None 表示该 dialogue 无有效数据。
    """
    if not turn_records:
        return None

    # 按 turn_idx 排序
    turn_records = sorted(turn_records, key=lambda x: x.get("turn_idx", 0))

    # 用最后一条 turn_record 的完整 history + 最后一个 assistant_reply 重建全局历史
    # 每条 record.history = 该 assistant turn 之前的所有 turn
    # record.assistant_reply = 该 assistant turn 的内容
    # 重建：取最后一条 record 的 history 加上最后一个 assistant_reply
    full_history: list[dict] = list(turn_records[-1]["history"])
    full_history.append({"role": "assistant", "content": turn_records[-1]["assistant_reply"]})

    # 但这样会丢失最后一个 assistant_reply 之后的 user turn。
    # 对于训练来说，我们只需要每个 assistant turn + 其历史，
    # 这些信息已经在每条 turn_record 中独立保存，可以直接使用。
    # 这里返回的 history 仅用于 preprocess_to_dict_data() 的迭代模式，
    # 因此直接从最完整的 turn_record 重建即可。

    satisfaction_scores = [r["gold_score"] for r in turn_records]
    reasons = [_score_to_reason(s) for s in satisfaction_scores]
    first = turn_records[0]

    return {
        "task":                  first.get("dataset", "USS"),
        "user":                  dialogue_id,
        "history":               full_history,
        "profile":               _DUMMY_PROFILE,
        "task_context":          first.get("task_context", ""),
        "satisfaction_scores":   satisfaction_scores,
        "dissatisfaction_reasons": reasons,
        "assistant_turns":       len(satisfaction_scores),
    }


def _build_flat_samples(turn_records: list[dict]) -> list[dict]:
    """
    逐 turn 展开：每条 record 直接转为一个 mini-sample（单个 assistant turn），
    供 preprocess_to_dict_data() 以对话级别以外的方式使用，
    或供自定义训练循环直接消费。

    返回 list of dict，每个 dict 有：
      persona, task_context, history (str), assistant_reply, score, reason, user
    """
    from lib.utils import conv_format  # 避免循环导入
    samples = []
    for r in turn_records:
        history_str = conv_format(r["history"])
        score = r["gold_score"]
        samples.append({
            "persona":         "",          # USS 无画像
            "task_context":    r.get("task_context", ""),
            "history":         history_str,
            "assistant_reply": r["assistant_reply"],
            "score":           score,
            "reason":          _score_to_reason(score),
            "user":            r["dialogue_id"],
        })
    return samples


def get_uss_data(
    datasets: list[str] | None = None,
    data_dir: str = _DEFAULT_USS_DIR,
    splits: list[str] | None = None,
) -> list[dict]:
    """
    加载 USS 预处理数据，返回与 get_satisfaction_data() 相同结构的 list[dict]。

    Args:
        datasets: 要加载的数据集列表，默认全部（JDDC/SGD/MWOZ/ReDial/CCPE）
        data_dir: 预处理 JSONL 文件目录（tools/preprocess_uss.py 的输出目录）
        splits:   要加载的 split，如 ["train"]、["val","test"]，默认全部

    Returns:
        list[dict]，每个 dict 结构与 get_satisfaction_data() 一致：
          task, user, history, profile, task_context,
          satisfaction_scores, dissatisfaction_reasons, assistant_turns
    """
    if datasets is None:
        datasets = ALL_DATASETS
    if splits is None:
        splits = ["train", "val", "test"]
    splits_set = set(splits)

    all_records: list[dict] = []
    for ds in datasets:
        path = os.path.join(data_dir, f"{ds}.jsonl")
        if not os.path.exists(path):
            logger.warning(f"USS processed file not found, skip: {path}")
            logger.warning("Run tools/preprocess_uss.py first.")
            continue
        records = _load_jsonl(path)
        filtered = [r for r in records if r.get("split") in splits_set]
        all_records.extend(filtered)
        logger.info(f"  {ds}: {len(filtered)} turn-records loaded (splits={splits})")

    # 按 dialogue 分组，重建 dialogue-level samples
    groups = _group_by_dialogue(all_records)
    samples: list[dict] = []
    for did, turns in groups.items():
        s = _build_sample(did, turns)
        if s is not None:
            samples.append(s)

    logger.info(f"USS data: {len(samples)} dialogue-level samples "
                f"({sum(s['assistant_turns'] for s in samples)} turn-level records)")
    return samples


def get_uss_flat_data(
    datasets: list[str] | None = None,
    data_dir: str = _DEFAULT_USS_DIR,
    splits: list[str] | None = None,
) -> list[dict]:
    """
    加载 USS 数据，返回展开的 turn-level dict 列表，
    格式与 preprocess_to_dict_data() 的输出兼容，可直接构建 Dataset。

    每个 dict 含：persona, task_context, history(str), assistant_reply,
                  score(int), reason(str), user(str)
    """
    if datasets is None:
        datasets = ALL_DATASETS
    if splits is None:
        splits = ["train", "val", "test"]
    splits_set = set(splits)

    all_records: list[dict] = []
    for ds in datasets:
        path = os.path.join(data_dir, f"{ds}.jsonl")
        if not os.path.exists(path):
            logger.warning(f"USS processed file not found, skip: {path}")
            continue
        records = _load_jsonl(path)
        filtered = [r for r in records if r.get("split") in splits_set]
        all_records.extend(filtered)

    flat = _build_flat_samples(all_records)
    logger.info(f"USS flat data: {len(flat)} turn-level samples (splits={splits})")
    return flat


if __name__ == "__main__":
    import argparse
    from collections import Counter

    parser = argparse.ArgumentParser(description="验证 USS 数据加载")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--data_dir", default=_DEFAULT_USS_DIR)
    parser.add_argument("--flat", action="store_true", help="使用 flat 模式")
    args = parser.parse_args()

    if args.flat:
        data = get_uss_flat_data(args.datasets, args.data_dir, args.splits)
        scores = [d["score"] for d in data]
        reasons = [d["reason"] for d in data]
        print(f"\n[Flat] total={len(data)}")
        print(f"  score dist: {dict(sorted(Counter(scores).items()))}")
        print(f"  reason dist: {dict(Counter(reasons))}")
        print(f"  sample: {json.dumps(data[0], ensure_ascii=False, indent=2)[:400]}")
    else:
        data = get_uss_data(args.datasets, args.data_dir, args.splits)
        all_scores = [s for d in data for s in d["satisfaction_scores"]]
        print(f"\n[Dialogue] total dialogues={len(data)}, total turns={len(all_scores)}")
        print(f"  score dist: {dict(sorted(Counter(all_scores).items()))}")
        print(f"  sample keys: {list(data[0].keys())}")
        print(f"  sample turns={data[0]['assistant_turns']}, task={data[0]['task']!r}")
