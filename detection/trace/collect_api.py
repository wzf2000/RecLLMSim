from __future__ import annotations

import os
import sys

_DETECTION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DETECTION_DIR not in sys.path:
    sys.path.insert(0, _DETECTION_DIR)

from trace.api_trace import (
    ReflectionAnswer,
    TraceAnswer,
    build_messages,
    build_prompt,
    build_reflection_user_feedback,
    collect_traces,
    dump_jsonl,
    generate_reflections_from_file,
    get_error_type,
    get_rows_from_split,
    load_finished_indices,
    load_jsonl,
    main,
    parse_args,
    predict_with_parse,
    preprocess_to_rows,
    reflect_with_parse,
)

__all__ = [
    "ReflectionAnswer",
    "TraceAnswer",
    "build_messages",
    "build_prompt",
    "build_reflection_user_feedback",
    "collect_traces",
    "dump_jsonl",
    "generate_reflections_from_file",
    "get_error_type",
    "get_rows_from_split",
    "load_finished_indices",
    "load_jsonl",
    "main",
    "parse_args",
    "predict_with_parse",
    "preprocess_to_rows",
    "reflect_with_parse",
]


if __name__ == "__main__":
    main()
