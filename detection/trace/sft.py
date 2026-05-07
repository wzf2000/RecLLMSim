"""Compatibility entrypoint for SFT data formatting and training."""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from .sft import (
        QWEN3_THINK_BEGIN,
        QWEN3_THINK_END,
        _BACKTICK_THINK_END,
        _THINK_VISIBLE_SPLIT_MARKERS,
        _balanced_json_spans,
        _row_first_attempt_wrong,
        _segment_looks_like_trace_json,
        _try_parse_trace_answer_obj,
        _unwrap_outer_code_fence,
        build_assistant_target,
        build_prompt_like_collect,
        build_source_text,
        filter_correct_records,
        filter_reflected_wrong_records,
        finalize_assistant_target_body,
        format_qwen3_think_wrapped_assistant_target,
        get_train_valid_dataset,
        load_jsonl,
        parse_args,
        parse_model_json,
        resolve_prompt_for_row,
        run_cli,
        select_records_for_sft,
        split_history_turns,
        strip_visible_answer_after_think,
        tokenize_example,
        train_sft,
    )
except ImportError:
    from sft import (
        QWEN3_THINK_BEGIN,
        QWEN3_THINK_END,
        _BACKTICK_THINK_END,
        _THINK_VISIBLE_SPLIT_MARKERS,
        _balanced_json_spans,
        _row_first_attempt_wrong,
        _segment_looks_like_trace_json,
        _try_parse_trace_answer_obj,
        _unwrap_outer_code_fence,
        build_assistant_target,
        build_prompt_like_collect,
        build_source_text,
        filter_correct_records,
        filter_reflected_wrong_records,
        finalize_assistant_target_body,
        format_qwen3_think_wrapped_assistant_target,
        get_train_valid_dataset,
        load_jsonl,
        parse_args,
        parse_model_json,
        resolve_prompt_for_row,
        run_cli,
        select_records_for_sft,
        split_history_turns,
        strip_visible_answer_after_think,
        tokenize_example,
        train_sft,
    )

__all__ = [
    "QWEN3_THINK_BEGIN",
    "QWEN3_THINK_END",
    "_BACKTICK_THINK_END",
    "_THINK_VISIBLE_SPLIT_MARKERS",
    "_balanced_json_spans",
    "_row_first_attempt_wrong",
    "_segment_looks_like_trace_json",
    "_try_parse_trace_answer_obj",
    "_unwrap_outer_code_fence",
    "build_assistant_target",
    "build_prompt_like_collect",
    "build_source_text",
    "filter_correct_records",
    "filter_reflected_wrong_records",
    "finalize_assistant_target_body",
    "format_qwen3_think_wrapped_assistant_target",
    "get_train_valid_dataset",
    "load_jsonl",
    "parse_args",
    "parse_model_json",
    "resolve_prompt_for_row",
    "run_cli",
    "select_records_for_sft",
    "split_history_turns",
    "strip_visible_answer_after_think",
    "tokenize_example",
    "train_sft",
]


if __name__ == "__main__":
    run_cli()
