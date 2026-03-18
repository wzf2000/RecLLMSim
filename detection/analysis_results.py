"""
分析 satisfaction 预测导出的结果 JSON。
支持：按 label 分数准确度、前一轮不满意后一轮准确度、各不满意 reason 准确率与预测分布等。
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr


def load_results(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def round_score(s: float) -> int:
    """预测分数四舍五入到整数（1-5）。"""
    return max(1, min(5, int(round(s))))


def get_task_from_path(file_path: str) -> str:
    """从 file_path 中解析任务名，如 .../User_0/菜谱规划/0.json -> 菜谱规划。"""
    parts = Path(file_path).parts
    if len(parts) >= 2:
        return parts[-2]  # .../TaskName/file.json
    return "unknown"


def analyze_by_label_score(results: list[dict]) -> dict:
    """不同 label 分数（1-5）对应的预测准确程度。"""
    by_score = defaultdict(list)
    for r in results:
        label = int(r["label_score"])
        pred = r["pred_score"]
        pred_reason = r["pred_reason"]
        label_reason = r["label_reason"]
        by_score[label].append({
            "pred_score": pred,
            "pred_round": round_score(pred),
            "label_reason": label_reason,
            "pred_reason": pred_reason,
        })

    out = {}
    for score in sorted(by_score.keys()):
        items = by_score[score]
        pred_scores = [x["pred_score"] for x in items]
        label_scores = [score] * len(items)
        # pred_round = [x["pred_round"] for x in items]
        reason_correct = [1 if x["label_reason"] == x["pred_reason"] else 0 for x in items]
        score_correct = [1 if round_score(p) == score else 0 for p in pred_scores]

        out[score] = {
            "count": len(items),
            "mae": mean_absolute_error(label_scores, pred_scores),
            "rmse": root_mean_squared_error(label_scores, pred_scores),
            "score_accuracy": np.mean(score_correct),
            "reason_accuracy": np.mean(reason_correct),
        }
    return out


def analyze_after_dissatisfied(results: list[dict]) -> dict:
    """前一轮 label<=3（不满意）时，当前轮的预测准确度。"""
    subset = []
    cnt = 0
    for r in results:
        turn = r["turn"]
        if turn <= 1:
            continue
        file_path = r["file_path"]
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        turn_cnt = 0
        prev_score = None
        for utt in data["history"]:
            if utt["role"] == "assistant":
                turn_cnt += 1
                if turn_cnt == turn - 1:
                    prev_score = int(utt["satisfaction"])
                    break
        assert prev_score is not None
        if prev_score <= 3:
            if r["label_score"] > 3:
                cnt += 1
            subset.append(r)
    print(f"不满意后变为满意的样本数: {cnt}")
    if not subset:
        return {"count": 0, "message": "无「前一轮不满意」的样本"}

    label_scores = [float(r["label_score"]) for r in subset]
    pred_scores = [r["pred_score"] for r in subset]
    reason_correct = [1 if r["label_reason"] == r["pred_reason"] else 0 for r in subset]
    score_round = [round_score(r["pred_score"]) for r in subset]
    score_correct = [1 if score_round[i] == int(label_scores[i]) else 0 for i in range(len(subset))]

    return {
        "count": len(subset),
        "mae": mean_absolute_error(label_scores, pred_scores),
        "rmse": root_mean_squared_error(label_scores, pred_scores),
        "pearson": pearsonr(label_scores, pred_scores)[0],
        "spearman": spearmanr(label_scores, pred_scores)[0],
        "score_accuracy": np.mean(score_correct),
        "reason_accuracy": np.mean(reason_correct),
    }


def analyze_label_calibration(results: list[dict]) -> dict:
    """
    对每个 label 分数：
    - 预测分数均值
    - 偏高/偏低/相等比例（基于 pred_score 与 label_score 的连续比较）
    """
    by_score = defaultdict(list)
    for r in results:
        by_score[int(r["label_score"])].append(float(r["pred_score"]))

    out = {}
    for score in sorted(by_score.keys()):
        preds = np.array(by_score[score], dtype=float)
        label = float(score)
        out[score] = {
            "count": int(preds.size),
            "pred_mean": float(preds.mean()) if preds.size else None,
            "pred_std": float(preds.std(ddof=0)) if preds.size else None,
            "higher_ratio": float(np.mean(preds > label)) if preds.size else None,
            "lower_ratio": float(np.mean(preds < label)) if preds.size else None,
            "equal_ratio": float(np.mean(preds == label)) if preds.size else None,
        }
    return out


def _binary_satisfaction_metrics(results: list[dict], threshold: float = 3.5) -> dict:
    """
    将 label 按 <=3 / >3 二分类（不满意/满意）。
    将 pred_score 按 threshold 二分类（pred>=threshold 认为满意）。
    """
    y_true = np.array([1 if float(r["label_score"]) > 3 else 0 for r in results], dtype=int)
    y_pred = np.array([1 if float(r["pred_score"]) >= threshold else 0 for r in results], dtype=int)
    if y_true.size == 0:
        return {"count": 0}
    return {
        "count": int(y_true.size),
        "threshold": float(threshold),
        "accuracy": float(np.mean(y_true == y_pred)),
        "pos_rate_label": float(y_true.mean()),
        "pos_rate_pred": float(y_pred.mean()),
        "pos_label_accuracy": float(np.mean(y_true[y_pred == 1] == 1)),
        "neg_label_accuracy": float(np.mean(y_true[y_pred == 0] == 0)),
    }


def analyze_binary_satisfaction(results: list[dict], threshold: float = 3.5) -> dict:
    """全局 + 前一轮不满意子集的二分类准确率（label: <=3/>3，pred: >=threshold）。"""
    overall = _binary_satisfaction_metrics(results, threshold=threshold)

    subset = []
    for r in results:
        turn = r["turn"]
        if turn <= 1:
            continue
        file_path = r["file_path"]
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        turn_cnt = 0
        prev_score = None
        for utt in data["history"]:
            if utt["role"] == "assistant":
                turn_cnt += 1
                if turn_cnt == turn - 1:
                    prev_score = int(utt["satisfaction"])
                    break
        assert prev_score is not None
        if prev_score <= 3:
            subset.append(r)
    after_prev_dissatisfied = _binary_satisfaction_metrics(subset, threshold=threshold)
    return {"overall": overall, "after_prev_dissatisfied": after_prev_dissatisfied}


def analyze_large_error_cases(results: list[dict], abs_diff_threshold: float = 2.0) -> dict:
    """
    预测与 GT 差距过大（|pred-label|>=threshold）的样本：
    - 样本量与占比
    - label 分数分布
    - （附加）按方向：预测偏高/偏低的比例
    """
    diffs = np.array([float(r["pred_score"]) - float(r["label_score"]) for r in results], dtype=float)
    mask = np.abs(diffs) >= abs_diff_threshold
    idx = np.where(mask)[0].tolist()
    if len(results) == 0:
        return {"count": 0, "ratio": 0.0, "threshold": abs_diff_threshold}

    label_dist = defaultdict(int)
    for i in idx:
        label_dist[int(float(results[i]["label_score"]))] += 1

    return {
        "threshold": float(abs_diff_threshold),
        "count": int(len(idx)),
        "ratio": float(len(idx) / len(results)),
        "label_distribution": dict(sorted(label_dist.items(), key=lambda x: x[0])),
        "higher_ratio": float(np.mean(diffs[mask] > 0)) if len(idx) else 0.0,
        "lower_ratio": float(np.mean(diffs[mask] < 0)) if len(idx) else 0.0,
    }


def analyze_pred_reason_score_alignment(results: list[dict], threshold: float = 3.5, satisfied_reason: str = "满意") -> dict:
    """
    reason 分类 与 满意度回归 的对齐程度（基于预测）：
    - pred_score>=threshold 时，pred_reason 是否为 satisfied_reason
    - pred_reason==satisfied_reason 时，pred_score 是否 >=threshold
    """
    if not results:
        return {"count": 0, "threshold": float(threshold), "satisfied_reason": satisfied_reason}

    pred_satisfied = np.array([float(r["pred_score"]) >= threshold for r in results], dtype=bool)
    pred_reason_satisfied = np.array([r["pred_reason"] == satisfied_reason for r in results], dtype=bool)

    n = len(results)
    n_pred_satisfied = int(pred_satisfied.sum())
    n_reason_satisfied = int(pred_reason_satisfied.sum())
    n_both = int(np.logical_and(pred_satisfied, pred_reason_satisfied).sum())

    # P(pred_reason==满意 | pred_score>=threshold)
    p_reason_given_score = float(n_both / n_pred_satisfied) if n_pred_satisfied else None
    # P(pred_score>=threshold | pred_reason==满意)
    p_score_given_reason = float(n_both / n_reason_satisfied) if n_reason_satisfied else None

    mismatch = np.logical_xor(pred_satisfied, pred_reason_satisfied)
    mismatch_rate = float(mismatch.mean())
    mismatch_breakdown = {
        "score_satisfied_but_reason_not": int(np.logical_and(pred_satisfied, ~pred_reason_satisfied).sum()),
        "reason_satisfied_but_score_not": int(np.logical_and(~pred_satisfied, pred_reason_satisfied).sum()),
    }

    return {
        "count": n,
        "threshold": float(threshold),
        "satisfied_reason": satisfied_reason,
        "n_pred_satisfied": n_pred_satisfied,
        "n_pred_reason_satisfied": n_reason_satisfied,
        "n_both_satisfied": n_both,
        "p_reason_satisfied_given_score_satisfied": p_reason_given_score,
        "p_score_satisfied_given_reason_satisfied": p_score_given_reason,
        "mismatch_rate": mismatch_rate,
        "mismatch_breakdown": mismatch_breakdown,
    }


def analyze_by_dissatisfaction_reason(results: list[dict]) -> dict:
    """仅对 label 不满意的样本：各 reason 的预测准确率和预测分布。"""
    dissatisfied = [r for r in results if float(r["label_score"]) <= 3]
    by_reason = defaultdict(list)
    for r in dissatisfied:
        by_reason[r["label_reason"]].append(r)

    out = {}
    all_reasons = set()
    for r in results:
        all_reasons.add(r["label_reason"])
        all_reasons.add(r["pred_reason"])
    all_reasons = sorted(all_reasons)

    for label_reason in sorted(by_reason.keys()):
        items = by_reason[label_reason]
        pred_reasons = [r["pred_reason"] for r in items]
        reason_correct = [1 if r["label_reason"] == r["pred_reason"] else 0 for r in items]
        pred_scores = [r["pred_score"] for r in items]
        label_scores = [float(r["label_score"]) for r in items]

        # 预测分布：各 pred_reason 的占比
        pred_dist = defaultdict(int)
        for pr in pred_reasons:
            pred_dist[pr] += 1
        pred_dist = {k: v / len(items) for k, v in pred_dist.items()}

        out[label_reason] = {
            "count": len(items),
            "reason_accuracy": np.mean(reason_correct),
            "mae": mean_absolute_error(label_scores, pred_scores),
            "prediction_distribution": pred_dist,
        }
    return out


def analyze_by_task(results: list[dict]) -> dict:
    """按任务类型（从 file_path 解析）的总体表现。"""
    by_task = defaultdict(list)
    for r in results:
        task = get_task_from_path(r["file_path"])
        by_task[task].append(r)

    out = {}
    for task in sorted(by_task.keys()):
        items = by_task[task]
        label_scores = [float(r["label_score"]) for r in items]
        pred_scores = [r["pred_score"] for r in items]
        reason_correct = [1 if r["label_reason"] == r["pred_reason"] else 0 for r in items]
        score_round = [round_score(r["pred_score"]) for r in items]
        score_correct = [1 if score_round[i] == int(label_scores[i]) else 0 for i in range(len(items))]

        out[task] = {
            "count": len(items),
            "mae": mean_absolute_error(label_scores, pred_scores),
            "score_accuracy": np.mean(score_correct),
            "reason_accuracy": np.mean(reason_correct),
        }
    return out


def analyze_by_turn(results: list[dict]) -> dict:
    """按对话轮次的预测表现（首轮 vs 多轮）。"""
    by_turn = defaultdict(list)
    for r in results:
        by_turn[r["turn"]].append(r)

    out = {}
    for turn in sorted(by_turn.keys()):
        items = by_turn[turn]
        label_scores = [float(r["label_score"]) for r in items]
        pred_scores = [r["pred_score"] for r in items]
        reason_correct = [1 if r["label_reason"] == r["pred_reason"] else 0 for r in items]
        out[turn] = {
            "count": len(items),
            "mae": mean_absolute_error(label_scores, pred_scores),
            "reason_accuracy": np.mean(reason_correct),
        }
    return out


def _get_chat_model_by_file_path(file_path: str, cache: dict[str, str]) -> str:
    """从 file_path 对应 JSON 中读取 chat_model，使用 cache 避免重复读文件。"""
    if file_path in cache:
        return cache[file_path]
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cache[file_path] = data.get("chat_model", "unknown")
    except (OSError, json.JSONDecodeError):
        cache[file_path] = "unknown"
    return cache[file_path]


def analyze_by_chat_model(results: list[dict]) -> dict:
    """
    按对话使用的 LLM（来自 file_path 对应 JSON 的 chat_model）分组：
    平均 label 分数、平均预测分数、label 分布（1-5）、预测分布（四舍五入到 1-5）。
    """
    cache: dict[str, str] = {}
    by_model = defaultdict(list)
    for r in results:
        model = _get_chat_model_by_file_path(r["file_path"], cache)
        by_model[model].append(r)

    out = {}
    for model in sorted(by_model.keys()):
        items = by_model[model]
        label_scores = [float(r["label_score"]) for r in items]
        pred_scores = [float(r["pred_score"]) for r in items]
        n = len(items)

        label_dist = defaultdict(int)
        for s in label_scores:
            label_dist[int(s)] += 1
        label_dist = dict(sorted(label_dist.items(), key=lambda x: x[0]))

        pred_rounded = [round_score(p) for p in pred_scores]
        pred_dist = defaultdict(int)
        for s in pred_rounded:
            pred_dist[s] += 1
        pred_dist = dict(sorted(pred_dist.items(), key=lambda x: x[0]))

        out[model] = {
            "count": n,
            "mean_label_score": float(np.mean(label_scores)),
            "mean_pred_score": float(np.mean(pred_scores)),
            "label_distribution": label_dist,
            "pred_distribution": pred_dist,
        }
    return out


def reason_confusion_matrix(results: list[dict]) -> dict:
    """Reason 预测的混淆统计：label_reason -> 预测成各 reason 的数量。"""
    label_list = [r["label_reason"] for r in results]
    pred_list = [r["pred_reason"] for r in results]
    all_reasons = sorted(set(label_list) | set(pred_list))
    cm = defaultdict(lambda: defaultdict(int))
    for label, pred in zip(label_list, pred_list):
        cm[label][pred] += 1
    return {"labels": all_reasons, "matrix": {k: dict(v) for k, v in cm.items()}}


def overall_metrics(results: list[dict]) -> dict:
    """整体指标。"""
    label_scores = [float(r["label_score"]) for r in results]
    pred_scores = [r["pred_score"] for r in results]
    label_reasons = [r["label_reason"] for r in results]
    pred_reasons = [r["pred_reason"] for r in results]
    score_round = [round_score(p) for p in pred_scores]
    score_correct = [1 if score_round[i] == int(label_scores[i]) else 0 for i in range(len(results))]

    return {
        "n_samples": len(results),
        "mae": mean_absolute_error(label_scores, pred_scores),
        "rmse": root_mean_squared_error(label_scores, pred_scores),
        "pearson": pearsonr(label_scores, pred_scores)[0],
        "spearman": spearmanr(label_scores, pred_scores)[0],
        "score_accuracy": np.mean(score_correct),
        "reason_accuracy": accuracy_score(label_reasons, pred_reasons),
        "reason_f1_weighted": f1_score(label_reasons, pred_reasons, average="weighted", zero_division=0),
    }


def print_section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="分析 satisfaction 预测导出的 JSON 结果")
    parser.add_argument("result_file", type=str, help="结果 JSON 路径，如 results/test_results_lora_0.json")
    parser.add_argument("--output", "-o", type=str, default="", help="将汇总结果写入 JSON；不指定则只打印")
    parser.add_argument("--no_print", action="store_true", help="不打印到终端，仅写文件")
    args = parser.parse_args()

    results = load_results(args.result_file)
    if not results:
        print("结果文件为空")
        return

    report = {
        "file": args.result_file,
        "n_total": len(results),
        "overall": overall_metrics(results),
        "by_label_score": analyze_by_label_score(results),
        "label_calibration": analyze_label_calibration(results),
        "binary_satisfaction": analyze_binary_satisfaction(results, threshold=3.5),
        "pred_reason_score_alignment": analyze_pred_reason_score_alignment(results, threshold=3.5, satisfied_reason="满意"),
        "large_error_cases": analyze_large_error_cases(results, abs_diff_threshold=2.0),
        "after_dissatisfied": analyze_after_dissatisfied(results),
        "by_dissatisfaction_reason": analyze_by_dissatisfaction_reason(results),
        "by_task": analyze_by_task(results),
        "by_turn": analyze_by_turn(results),
        "by_chat_model": analyze_by_chat_model(results),
        "reason_confusion": reason_confusion_matrix(results),
    }

    def do_print(text: str):
        if not args.no_print:
            print(text)

    do_print(f"\n结果文件: {args.result_file}  样本数: {len(results)}")

    print_section("整体指标")
    for k, v in report["overall"].items():
        do_print(f"  {k}: {v}")

    print_section("按 label 分数 (1-5) 的预测表现")
    for score, m in report["by_label_score"].items():
        do_print(f"  Label {score}: n={m['count']}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  score_acc={m['score_accuracy']:.4f}  reason_acc={m['reason_accuracy']:.4f}")

    print_section("每个 label 分数的预测均值 & 偏高/偏低比例")
    for score, m in report["label_calibration"].items():
        do_print(
            f"  Label {score}: n={m['count']}  pred_mean={m['pred_mean']:.4f}  pred_std={m['pred_std']:.4f}  "
            f"higher={m['higher_ratio']:.4f}  lower={m['lower_ratio']:.4f}  equal={m['equal_ratio']:.4f}"
        )

    print_section("满意/不满意二分类准确率（label:<=3/>3, pred>=3.5 视为满意）")
    bs = report["binary_satisfaction"]
    do_print(
        f"  [Overall] n={bs['overall']['count']}  acc={bs['overall']['accuracy']:.4f}  "
        f"label_pos_rate={bs['overall']['pos_rate_label']:.4f}  pred_pos_rate={bs['overall']['pos_rate_pred']:.4f}  "
        f"pos_label_accuracy={bs['overall']['pos_label_accuracy']:.4f}  neg_label_accuracy={bs['overall']['neg_label_accuracy']:.4f}"
    )
    do_print(
        f"  [After prev dissatisfied] n={bs['after_prev_dissatisfied']['count']}  "
        f"acc={(bs['after_prev_dissatisfied'].get('accuracy') if bs['after_prev_dissatisfied']['count'] else 0.0):.4f}  "
        f"label_pos_rate={(bs['after_prev_dissatisfied'].get('pos_rate_label') if bs['after_prev_dissatisfied']['count'] else 0.0):.4f}  "
        f"pred_pos_rate={(bs['after_prev_dissatisfied'].get('pos_rate_pred') if bs['after_prev_dissatisfied']['count'] else 0.0):.4f}  "
        f"pos_label_accuracy={bs['after_prev_dissatisfied'].get('pos_label_accuracy') if bs['after_prev_dissatisfied']['count'] else 0.0:.4f}   "
        f"neg_label_accuracy={bs['after_prev_dissatisfied'].get('neg_label_accuracy') if bs['after_prev_dissatisfied']['count'] else 0.0:.4f}"
    )

    print_section("reason分类 与 满意度回归 的对齐程度（基于预测）")
    al = report["pred_reason_score_alignment"]
    do_print(f"  threshold={al['threshold']}  satisfied_reason={al['satisfied_reason']}  n={al['count']}")
    do_print(
        f"  P(pred_reason=满意 | pred_score>=thr)={al['p_reason_satisfied_given_score_satisfied']}  "
        f"(n_score_pos={al['n_pred_satisfied']}, n_both={al['n_both_satisfied']})"
    )
    do_print(
        f"  P(pred_score>=thr | pred_reason=满意)={al['p_score_satisfied_given_reason_satisfied']}  "
        f"(n_reason_pos={al['n_pred_reason_satisfied']}, n_both={al['n_both_satisfied']})"
    )
    do_print(f"  mismatch_rate={al['mismatch_rate']:.4f}  mismatch_breakdown={al['mismatch_breakdown']}")

    print_section("前一轮不满意 (label≤3) 后一轮的预测准确度")
    ad = report["after_dissatisfied"]
    if ad.get("count", 0) == 0:
        do_print("  " + ad.get("message", "无数据"))
    else:
        do_print(f"  样本数: {ad['count']}")
        do_print(f"  MAE={ad['mae']:.4f}  RMSE={ad['rmse']:.4f}  Pearson={ad['pearson']:.4f}  Spearman={ad['spearman']:.4f}")
        do_print(f"  score_accuracy={ad['score_accuracy']:.4f}  reason_accuracy={ad['reason_accuracy']:.4f}")

    print_section("各不满意 reason 的准确率与预测分布")
    for reason, m in report["by_dissatisfaction_reason"].items():
        do_print(f"  [{reason}] n={m['count']}  reason_accuracy={m['reason_accuracy']:.4f}  MAE={m['mae']:.4f}")
        do_print(f"    预测分布: {m['prediction_distribution']}")

    print_section("按任务类型")
    for task, m in report["by_task"].items():
        do_print(f"  {task}: n={m['count']}  MAE={m['mae']:.4f}  score_acc={m['score_accuracy']:.4f}  reason_acc={m['reason_accuracy']:.4f}")

    print_section("按对话使用的 LLM (chat_model)")
    for model, m in report["by_chat_model"].items():
        do_print(f"  [{model}] n={m['count']}  mean_label={m['mean_label_score']:.4f}  mean_pred={m['mean_pred_score']:.4f}")
        do_print(f"    label_dist={m['label_distribution']}  pred_dist={m['pred_distribution']}")

    print_section("按对话轮次 (前几轮)")
    turns = sorted(report["by_turn"].keys())[:10]
    for turn in turns:
        m = report["by_turn"][turn]
        do_print(f"  turn={turn}: n={m['count']}  MAE={m['mae']:.4f}  reason_acc={m['reason_accuracy']:.4f}")
    if len(report["by_turn"]) > 10:
        do_print(f"  ... 共 {len(report['by_turn'])} 个轮次")

    print_section("Reason 混淆统计 (label -> 预测)")
    cm = report["reason_confusion"]
    do_print("  labels: " + ", ".join(cm["labels"]))
    for label in cm["labels"]:
        if label in cm["matrix"]:
            do_print(f"  {label} -> {dict(cm['matrix'][label])}")

    print_section("大误差样本（|pred-label|>=2）标签分布")
    le = report["large_error_cases"]
    do_print(f"  threshold={le['threshold']}  count={le['count']}  ratio={le['ratio']:.4f}")
    do_print(f"  label_distribution={le['label_distribution']}")
    do_print(f"  higher_ratio={le['higher_ratio']:.4f}  lower_ratio={le['lower_ratio']:.4f}")

    if args.output:
        out_path = args.output
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        do_print(f"\n汇总已写入: {out_path}")


if __name__ == "__main__":
    main()
