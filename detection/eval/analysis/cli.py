from __future__ import annotations

import argparse
import json
from pathlib import Path

from .binary_metrics import analyze_binary_satisfaction
from .diagnostics import analyze_large_error_cases, analyze_pred_reason_score_alignment
from .grouped import analyze_by_chat_model, analyze_by_dissatisfaction_reason, analyze_by_task, analyze_by_turn
from .io import load_results
from .overall import overall_metrics, reason_confusion_matrix
from .score_metrics import analyze_after_dissatisfied, analyze_by_label_score, analyze_label_calibration


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


