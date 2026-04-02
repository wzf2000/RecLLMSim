"""
USS (User Satisfaction Simulation) 数据集预处理脚本。
论文：Simulating User Satisfaction for the Evaluation of Task-oriented Dialogue Systems
      (Sun et al., SIGIR 2021) https://dl.acm.org/doi/10.1145/3404835.3463241
数据：https://github.com/sunnweiwei/user-satisfaction-simulation

格式说明：
  - 每行 tab 分隔：role, text, action, scores
  - scores 为多标注者逗号分隔的 1-5 整数（SYSTEM 行为空）
  - session 间用空行分隔
  - 每个 session 最后一行为 USER / OVERALL（整体评分），文本为 "OVERALL"

预处理逻辑：
  - 每个 SYSTEM turn 对应一条训练样本
  - assistant_reply = 该 SYSTEM turn 的文本
  - history = 该 SYSTEM turn 之前所有 turn
  - gold_score = 该 SYSTEM turn 之后 USER turn 的多标注众数分
  - OVERALL turn 仅用于记录 session 整体分，不生成 turn-level 样本

用法：
  python tools/preprocess_uss.py
  python tools/preprocess_uss.py --raw_dir data/uss/raw --out_dir data/uss/processed
"""

import json
import os
import argparse
from collections import Counter
from pathlib import Path


# 数据集元信息
DATASETS = {
    "JDDC":   {"file": "JDDC.txt",   "lang": "zh", "domain": "Customer Service"},
    "SGD":    {"file": "SGD.txt",    "lang": "en", "domain": "Task-oriented (Multi-domain)"},
    "MWOZ":   {"file": "MWOZ.txt",   "lang": "en", "domain": "Task-oriented (Hotel/Restaurant/Taxi)"},
    "ReDial": {"file": "ReDial.txt", "lang": "en", "domain": "Movie Recommendation"},
    "CCPE":   {"file": "CCPE.txt",   "lang": "en", "domain": "Preference Elicitation"},
}

DUMMY_PROFILE = {
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


def get_mode(scores: list[int]) -> int:
    """取众数，平局时取较大值。"""
    return Counter(scores).most_common(1)[0][0]


def parse_session(lines: list[str]) -> list[dict]:
    """解析一个 session（空行分隔），返回 turn 列表。"""
    turns = []
    for line in lines:
        if not line.strip():
            continue
        # 注意：不对整行 strip，否则尾部空 tab（JDDC/ReDial SYSTEM 行无 action/scores）
        # 会被吃掉导致 field 数量变少
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        role = parts[0].strip()
        text = parts[1].strip() if len(parts) > 1 else ""
        action = parts[2].strip() if len(parts) > 2 else ""
        raw_scores = parts[3].strip() if len(parts) > 3 else ""
        scores = []
        if raw_scores:
            try:
                scores = [int(s) for s in raw_scores.split(",") if s.strip()]
            except ValueError:
                pass
        turns.append({"role": role, "text": text, "action": action, "scores": scores})
    return turns


def session_to_samples(
    session: list[dict],
    dialogue_id: str,
    dataset: str,
) -> tuple[list[dict], dict | None]:
    """
    将一个对话 session 拆成 turn-level 样本。

    返回 (turn_samples, overall_record)
    - turn_samples: 每个 SYSTEM turn 一条
    - overall_record: OVERALL 整体评分（可能为 None）
    """
    turn_samples: list[dict] = []
    overall: dict | None = None

    # 建立 role → history index 映射，找 SYSTEM turn 后接的 USER turn scores
    # history_so_far 在 SYSTEM turn 之前的所有 turn
    history_so_far: list[dict] = []    # list of {role, content}
    system_turn_buf: dict | None = None  # 还未拿到 score 的 SYSTEM turn

    system_turn_idx = 0

    for i, turn in enumerate(session):
        role = turn["role"]
        text = turn["text"]
        scores = turn["scores"]

        if role == "USER" and text == "OVERALL":
            # session 级整体评分，单独记录
            if scores:
                overall = {
                    "dialogue_id": dialogue_id,
                    "dataset": dataset,
                    "overall_score": get_mode(scores),
                    "overall_score_raw": scores,
                }
            continue

        if role == "SYSTEM":
            # 如果还有上一个 SYSTEM turn 没拿到 USER score，先保存（用默认分 3）
            if system_turn_buf is not None:
                turn_samples.append(system_turn_buf)
            system_turn_buf = {
                "dialogue_id": dialogue_id,
                "dataset": dataset,
                "turn_idx": system_turn_idx,
                "history": list(history_so_far),
                "assistant_reply": text,
                "action": turn["action"],
                "gold_score": None,       # 等待下一个 USER turn 填充
                "gold_score_raw": None,
            }
            system_turn_idx += 1
            history_so_far.append({"role": "assistant", "content": text})

        elif role == "USER":
            if system_turn_buf is not None and scores:
                # 用该 USER turn 的 scores 作为前一个 SYSTEM turn 的满意度
                system_turn_buf["gold_score"] = get_mode(scores)
                system_turn_buf["gold_score_raw"] = scores
                turn_samples.append(system_turn_buf)
                system_turn_buf = None
            elif system_turn_buf is not None and not scores:
                # USER turn 无 scores（少见），跳过，SYSTEM turn 也丢弃
                system_turn_buf = None
            # 无论如何把 USER turn 加入 history
            history_so_far.append({"role": "user", "content": text})

    # 末尾残留的 SYSTEM turn（无后续 USER 打分），丢弃
    return turn_samples, overall


def parse_file(
    filepath: str,
    dataset: str,
) -> tuple[list[dict], list[dict]]:
    """解析单个 USS .txt 文件，返回 (turn_samples, overall_records)。"""
    turn_samples: list[dict] = []
    overall_records: list[dict] = []

    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    raw_sessions = content.strip().split("\n\n")
    for sess_idx, raw_sess in enumerate(raw_sessions):
        lines = raw_sess.strip().split("\n")
        session = parse_session(lines)
        if not session:
            continue
        dialogue_id = f"{dataset}_{sess_idx:05d}"
        samples, overall = session_to_samples(session, dialogue_id, dataset)
        turn_samples.extend(samples)
        if overall:
            overall_records.append(overall)

    return turn_samples, overall_records


def add_reason(sample: dict) -> dict:
    """根据 gold_score 填充简化 reason（与项目 REASON_TO_ID 对齐）。"""
    score = sample.get("gold_score")
    if score is None:
        sample["reason"] = "其它"
    elif score >= 4:
        sample["reason"] = "满意"
    else:
        sample["reason"] = "其它"   # USS 无细粒度原因，统一用"其它"
    return sample


def add_split(samples: list[dict], seed: int = 42) -> list[dict]:
    """
    按 dialogue_id 分组做 8:1:1 train/val/test 划分。
    同一 dialogue 的所有 turn 不跨集合。
    """
    import random
    rng = random.Random(seed)
    dialogues = sorted(set(s["dialogue_id"] for s in samples))
    rng.shuffle(dialogues)
    n = len(dialogues)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    train_set = set(dialogues[:n_train])
    val_set   = set(dialogues[n_train:n_train + n_val])
    for s in samples:
        if s["dialogue_id"] in train_set:
            s["split"] = "train"
        elif s["dialogue_id"] in val_set:
            s["split"] = "val"
        else:
            s["split"] = "test"
    return samples


def save_jsonl(records: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):,} records → {path}")


def print_stats(samples: list[dict], dataset: str) -> None:
    scores = [s["gold_score"] for s in samples if s.get("gold_score") is not None]
    dist = Counter(scores)
    n_dlg = len(set(s["dialogue_id"] for s in samples))
    print(f"\n  [{dataset}]  dialogues={n_dlg}  turns={len(samples)}")
    print(f"    score dist: " + "  ".join(f"{k}:{dist[k]}" for k in sorted(dist)))
    splits = Counter(s["split"] for s in samples)
    print(f"    split: train={splits['train']}  val={splits['val']}  test={splits['test']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/uss/raw",
                        help="原始 USS .txt 文件目录")
    parser.add_argument("--out_dir", default="data/uss/processed",
                        help="输出 JSONL 目录")
    parser.add_argument("--datasets", nargs="+",
                        default=list(DATASETS.keys()),
                        choices=list(DATASETS.keys()),
                        help="要处理的数据集（默认全部）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_turn_samples: list[dict] = []
    all_overall_records: list[dict] = []

    for ds_name in args.datasets:
        info = DATASETS[ds_name]
        filepath = os.path.join(args.raw_dir, info["file"])
        if not os.path.exists(filepath):
            print(f"[WARN] 文件不存在，跳过: {filepath}")
            continue

        print(f"\nParsing {ds_name} ({info['lang']}, {info['domain']}) ...")
        turn_samples, overall_records = parse_file(filepath, ds_name)

        # 过滤掉没有 gold_score 的样本
        turn_samples = [s for s in turn_samples if s.get("gold_score") is not None]

        # 填充 reason、profile 占位、task_context
        for s in turn_samples:
            add_reason(s)
            s["lang"]         = info["lang"]
            s["task_context"] = f"{ds_name} - {info['domain']}"
            s["profile"]      = DUMMY_PROFILE

        # 划分 split
        add_split(turn_samples, seed=args.seed)

        # 输出每个数据集单独文件
        save_jsonl(turn_samples, os.path.join(args.out_dir, f"{ds_name}.jsonl"))
        if overall_records:
            save_jsonl(overall_records, os.path.join(args.out_dir, f"{ds_name}_overall.jsonl"))

        print_stats(turn_samples, ds_name)
        all_turn_samples.extend(turn_samples)
        all_overall_records.extend(overall_records)

    if len(args.datasets) > 1 and all_turn_samples:
        save_jsonl(all_turn_samples, os.path.join(args.out_dir, "all.jsonl"))
        print_stats(all_turn_samples, "ALL")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
