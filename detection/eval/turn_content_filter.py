"""
LLM-based turn content annotation and grouped evaluation.

This compatibility entrypoint keeps the historical commands working:

  python eval/turn_content_filter.py annotate ...
  python eval/turn_content_filter.py analyze ...
"""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from .turn_content.analysis import _metrics, _parse_named_files, _sat, command_analyze
    from .turn_content.annotation import (
        TargetTurn,
        TurnContentAnnotation,
        _current_turn_exchange,
        _extract_json_object,
        _iter_target_turns,
        _message_content_to_text,
        build_annotation_prompt,
        call_annotation,
        command_annotate,
        load_existing_annotations,
        load_target_turns,
        set_client,
    )
    from .turn_content.cli import build_parser, main
    from .turn_content.io import load_jsonl, save_jsonl
except ImportError:
    from turn_content.analysis import _metrics, _parse_named_files, _sat, command_analyze
    from turn_content.annotation import (
        TargetTurn,
        TurnContentAnnotation,
        _current_turn_exchange,
        _extract_json_object,
        _iter_target_turns,
        _message_content_to_text,
        build_annotation_prompt,
        call_annotation,
        command_annotate,
        load_existing_annotations,
        load_target_turns,
        set_client,
    )
    from turn_content.cli import build_parser, main
    from turn_content.io import load_jsonl, save_jsonl

__all__ = [
    "TargetTurn",
    "TurnContentAnnotation",
    "_current_turn_exchange",
    "_extract_json_object",
    "_iter_target_turns",
    "_message_content_to_text",
    "_metrics",
    "_parse_named_files",
    "_sat",
    "build_annotation_prompt",
    "build_parser",
    "call_annotation",
    "command_analyze",
    "command_annotate",
    "load_existing_annotations",
    "load_jsonl",
    "load_target_turns",
    "main",
    "save_jsonl",
    "set_client",
]


if __name__ == "__main__":
    main()
