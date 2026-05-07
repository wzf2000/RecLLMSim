"""
SPUR: Supervised Prompting for User satisfaction Rubrics
参考: Lin et al., ACL 2024 (arXiv:2403.12388)

Compatibility entrypoint for the historical command:

  python eval/spur.py ...
"""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from .spur import (
        DSAT_LABEL,
        SAT_LABEL,
        _EXTRACT_DSAT_TMPL,
        _EXTRACT_SAT_TMPL,
        _EXTRACT_SYSTEM,
        _MAX_EMBEDDING_TOKENS,
        _SCORING_SYSTEM,
        _SCORING_TMPL,
        _SUMMARIZE_DSAT_TMPL,
        _SUMMARIZE_SAT_TMPL,
        _SUMMARIZE_SYSTEM,
        _call_llm,
        _embed_batch,
        _extract_rubrics_for_one,
        _format_rubrics,
        _get_tiktoken_enc,
        _score_one,
        _summarize_one_label,
        _sys_user,
        _truncate_text,
        build_rubric_feature_vec,
        compute_metrics,
        extract_rubric_candidates,
        format_conversation,
        get_embeddings,
        main,
        parse_args,
        preprocess_to_rows,
        print_metrics,
        run_cli,
        score_rows,
        score_test_set,
        summarize_rubrics,
        train_and_eval_classifier,
    )
except ImportError:
    from spur import (
        DSAT_LABEL,
        SAT_LABEL,
        _EXTRACT_DSAT_TMPL,
        _EXTRACT_SAT_TMPL,
        _EXTRACT_SYSTEM,
        _MAX_EMBEDDING_TOKENS,
        _SCORING_SYSTEM,
        _SCORING_TMPL,
        _SUMMARIZE_DSAT_TMPL,
        _SUMMARIZE_SAT_TMPL,
        _SUMMARIZE_SYSTEM,
        _call_llm,
        _embed_batch,
        _extract_rubrics_for_one,
        _format_rubrics,
        _get_tiktoken_enc,
        _score_one,
        _summarize_one_label,
        _sys_user,
        _truncate_text,
        build_rubric_feature_vec,
        compute_metrics,
        extract_rubric_candidates,
        format_conversation,
        get_embeddings,
        main,
        parse_args,
        preprocess_to_rows,
        print_metrics,
        run_cli,
        score_rows,
        score_test_set,
        summarize_rubrics,
        train_and_eval_classifier,
    )

__all__ = [
    "DSAT_LABEL",
    "SAT_LABEL",
    "_EXTRACT_DSAT_TMPL",
    "_EXTRACT_SAT_TMPL",
    "_EXTRACT_SYSTEM",
    "_MAX_EMBEDDING_TOKENS",
    "_SCORING_SYSTEM",
    "_SCORING_TMPL",
    "_SUMMARIZE_DSAT_TMPL",
    "_SUMMARIZE_SAT_TMPL",
    "_SUMMARIZE_SYSTEM",
    "_call_llm",
    "_embed_batch",
    "_extract_rubrics_for_one",
    "_format_rubrics",
    "_get_tiktoken_enc",
    "_score_one",
    "_summarize_one_label",
    "_sys_user",
    "_truncate_text",
    "build_rubric_feature_vec",
    "compute_metrics",
    "extract_rubric_candidates",
    "format_conversation",
    "get_embeddings",
    "main",
    "parse_args",
    "preprocess_to_rows",
    "print_metrics",
    "run_cli",
    "score_rows",
    "score_test_set",
    "summarize_rubrics",
    "train_and_eval_classifier",
]


if __name__ == "__main__":
    run_cli()
