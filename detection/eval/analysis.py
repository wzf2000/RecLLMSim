"""Compatibility entrypoint for satisfaction result diagnostics."""

from __future__ import annotations

import os
import sys

_DETECTION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DETECTION_DIR not in sys.path:
    sys.path.insert(0, _DETECTION_DIR)

from eval.analysis import (
    _binary_satisfaction_metrics,
    _get_chat_model_by_file_path,
    analyze_after_dissatisfied,
    analyze_binary_satisfaction,
    analyze_by_chat_model,
    analyze_by_dissatisfaction_reason,
    analyze_by_label_score,
    analyze_by_task,
    analyze_by_turn,
    analyze_label_calibration,
    analyze_large_error_cases,
    analyze_pred_reason_score_alignment,
    get_task_from_path,
    load_results,
    main,
    overall_metrics,
    print_section,
    reason_confusion_matrix,
    round_score,
)

__all__ = [
    "_binary_satisfaction_metrics",
    "_get_chat_model_by_file_path",
    "analyze_after_dissatisfied",
    "analyze_binary_satisfaction",
    "analyze_by_chat_model",
    "analyze_by_dissatisfaction_reason",
    "analyze_by_label_score",
    "analyze_by_task",
    "analyze_by_turn",
    "analyze_label_calibration",
    "analyze_large_error_cases",
    "analyze_pred_reason_score_alignment",
    "get_task_from_path",
    "load_results",
    "main",
    "overall_metrics",
    "print_section",
    "reason_confusion_matrix",
    "round_score",
]


if __name__ == "__main__":
    main()
