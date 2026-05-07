from .cli import parse_args, run_cli
from .dataset import get_train_valid_dataset, tokenize_example
from .parsing import (
    QWEN3_THINK_BEGIN,
    QWEN3_THINK_END,
    _BACKTICK_THINK_END,
    _THINK_VISIBLE_SPLIT_MARKERS,
    _balanced_json_spans,
    _segment_looks_like_trace_json,
    _try_parse_trace_answer_obj,
    _unwrap_outer_code_fence,
    parse_model_json,
    strip_visible_answer_after_think,
)
from .prompts import (
    build_prompt_like_collect,
    build_source_text,
    resolve_prompt_for_row,
    split_history_turns,
)
from .selection import (
    _row_first_attempt_wrong,
    filter_correct_records,
    filter_reflected_wrong_records,
    load_jsonl,
    select_records_for_sft,
)
from .targets import (
    build_assistant_target,
    finalize_assistant_target_body,
    format_qwen3_think_wrapped_assistant_target,
)
from .training import train_sft

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
