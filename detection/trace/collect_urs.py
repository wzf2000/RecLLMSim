"""Compatibility entrypoint for URS session-level personalized collection."""

from __future__ import annotations

import os
import sys

_DETECTION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DETECTION_DIR not in sys.path:
    sys.path.insert(0, _DETECTION_DIR)

from trace.urs import (
    MemoryUpdateMode,
    _call_build_memory,
    _call_predict_session,
    _call_update_memory,
    _normalize_pred_reason,
    build_user_memory_urs,
    collect_all_urs,
    evaluate_urs_session,
    load_finished_ids,
    main,
    parse_args,
    run_agent_on_urs_sample,
    update_memory_urs,
)

__all__ = [
    "MemoryUpdateMode",
    "_call_build_memory",
    "_call_predict_session",
    "_call_update_memory",
    "_normalize_pred_reason",
    "build_user_memory_urs",
    "collect_all_urs",
    "evaluate_urs_session",
    "load_finished_ids",
    "main",
    "parse_args",
    "run_agent_on_urs_sample",
    "update_memory_urs",
]


if __name__ == "__main__":
    main()
