"""Streamlit app for pairwise human validation of replayed responses."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

import streamlit as st


PREFERENCE_OPTIONS = {
    "A is better": "a",
    "B is better": "b",
    "Tie / similar": "tie",
    "Cannot judge": "uncertain",
}

CONFIDENCE_OPTIONS = ["high", "medium", "low"]

REASON_OPTIONS = [
    "better addresses the user request",
    "more personalized",
    "more specific or actionable",
    "better follows constraints",
    "better structure or readability",
    "more reliable or safer",
    "less verbose",
    "other",
]

SUBSET_OPTIONS = {
    "all": "All items",
    "first_half": "First half",
    "second_half": "Second half",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise replay validation app.")
    parser.add_argument(
        "--items_jsonl",
        default="outputs/human_validation/replay_pairwise_items.jsonl",
        help="Pairwise item JSONL produced by tools/build_replay_pairwise_items.py.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/human_validation/annotations",
        help="Directory for per-annotator JSONL files.",
    )
    parser.add_argument("--page_title", default="Replay Pairwise Validation")
    parser.add_argument("--show_debug_default", action="store_true")
    args, _ = parser.parse_known_args()
    return args


@st.cache_data(show_spinner=False)
def load_items(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def safe_annotator_id(text: str) -> str:
    out = (text or "").strip()
    for sep in [os.sep, getattr(os.path, "altsep", None)]:
        if sep:
            out = out.replace(sep, "_")
    return out.replace(" ", "_")


def annotation_path(output_dir: str, annotator_id: str) -> str:
    return os.path.join(output_dir, f"{safe_annotator_id(annotator_id)}.jsonl")


def load_annotations(path: str) -> dict[str, dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    by_item: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            item_id = str(record.get("item_id", ""))
            if item_id:
                by_item[item_id] = record
    return by_item


def append_annotation(path: str, annotation: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(annotation, ensure_ascii=False) + "\n")


def first_unannotated_index(items: list[dict[str, Any]], annotations: dict[str, dict[str, Any]]) -> int:
    annotated_ids = set(annotations)
    for idx, item in enumerate(items):
        if str(item.get("item_id")) not in annotated_ids:
            return idx
    return len(items)


def subset_bounds(total: int, subset_key: str) -> tuple[int, int]:
    midpoint = (total + 1) // 2
    if subset_key == "first_half":
        return 0, midpoint
    if subset_key == "second_half":
        return midpoint, total
    return 0, total


def subset_caption(subset_key: str, start: int, end: int, total: int) -> str:
    if total <= 0:
        return "No items"
    if subset_key == "all":
        return f"All items: 1-{total}"
    return f"{SUBSET_OPTIONS[subset_key]}: {start + 1}-{end} of {total}"


def format_profile(profile: dict[str, Any]) -> None:
    if not profile:
        st.caption("No profile is available for this item.")
        return
    field_names = [
        ("gender", "Gender"),
        ("age", "Age"),
        ("occupation", "Occupation"),
        ("background", "Background"),
        ("personality", "Personality"),
        ("daily_interests", "Daily interests"),
        ("travel_habits", "Travel habits"),
        ("dining_preferences", "Dining preferences"),
        ("spending_habits", "Spending habits"),
        ("other_aspects", "Other aspects"),
    ]
    for key, label in field_names:
        value = profile.get(key)
        if value in (None, "", []):
            continue
        if isinstance(value, list):
            value_text = ", ".join(str(v) for v in value)
        else:
            value_text = str(value)
        st.markdown(f"**{label}:** {value_text}")


def render_score_distribution(dist: dict[str, Any]) -> None:
    if not dist or not dist.get("total_turns"):
        st.caption("No historical score distribution is available.")
        return
    counts = dist.get("counts", {})
    rows = [
        {
            "Score": score,
            "Count": int(counts.get(str(score), 0)),
        }
        for score in range(1, 6)
    ]
    st.markdown(
        f"**Historical mean:** {float(dist.get('mean', 0.0)):.2f} "
        f"over {int(dist.get('total_turns', 0))} source-history assistant turns"
    )
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.caption(
        "SAT = scores 4-5; Neutral = score 3; low = scores 1-2. "
        f"SAT rate {float(dist.get('sat_rate') or 0.0):.1%}, "
        f"Neutral rate {float(dist.get('neutral_rate') or 0.0):.1%}, "
        f"low-score rate {float(dist.get('low_rate') or 0.0):.1%}."
    )


def render_anchor_example(example: dict[str, Any], idx: int) -> None:
    title = (
        f"{idx}. {example.get('source_task', '')} / "
        f"{example.get('source_file', '')} / turn {example.get('turn_idx', '')} "
        f"/ score {example.get('score', '')}"
    )
    with st.expander(title, expanded=False):
        reason = example.get("reason")
        if reason:
            st.markdown(f"**Historical reason:** {reason}")
        st.markdown("**Historical user request**")
        st.write(example.get("user_request", ""))
        st.markdown("**Historical assistant response**")
        st.write(example.get("assistant_response", ""))


def render_user_preference_evidence(evidence: dict[str, Any]) -> None:
    if not evidence:
        st.caption("No source-history preference evidence is available for this item.")
        return

    st.caption(
        "Generated from this user's labeled source-history conversations in other scenarios. "
        "It does not include the current hidden evaluator score or model identity."
    )
    summary = evidence.get("summary", [])
    if summary:
        st.markdown("**Preference summary**")
        for bullet in summary:
            st.markdown(f"- {bullet}")

    render_score_distribution(evidence.get("score_distribution", {}))

    reasons = evidence.get("low_side_reasons", [])
    if reasons:
        st.markdown("**Common low-side / neutral reasons**")
        reason_rows = [
            {"Reason": item.get("reason", ""), "Count": int(item.get("count", 0))}
            for item in reasons
        ]
        st.dataframe(reason_rows, hide_index=True, use_container_width=True)

    anchors = evidence.get("anchor_examples", {})
    high_examples = anchors.get("high_score", [])
    low_examples = anchors.get("low_or_neutral", [])
    if high_examples or low_examples:
        st.markdown("**Historical anchor examples**")
        high_col, low_col = st.columns(2)
        with high_col:
            st.markdown("High-score examples")
            if not high_examples:
                st.caption("No high-score anchors.")
            for idx, example in enumerate(high_examples, 1):
                render_anchor_example(example, idx)
        with low_col:
            st.markdown("Low / neutral examples")
            if not low_examples:
                st.caption("No low/neutral anchors.")
            for idx, example in enumerate(low_examples, 1):
                render_anchor_example(example, idx)


def render_conversation(prefix: list[dict[str, Any]]) -> None:
    if not prefix:
        st.caption("No conversation prefix is available.")
        return
    with st.container(height=460):
        for utt in prefix:
            role = "user" if utt.get("role") == "user" else "assistant"
            with st.chat_message(role):
                st.write(utt.get("content", ""))


def render_response(label: str, side: dict[str, Any]) -> None:
    st.subheader(f"Response {label}")
    with st.container(height=460):
        st.markdown(side.get("response", "") or "_Empty response._")


def human_choice_to_response_id(item: dict[str, Any], choice: str) -> str:
    if choice == "a":
        return str(item.get("side_a", {}).get("response_id", ""))
    if choice == "b":
        return str(item.get("side_b", {}).get("response_id", ""))
    return ""


def agreement_with_evaluator(item: dict[str, Any], human_choice: str) -> bool | None:
    evaluator_choice = str(item.get("evaluator_preference", ""))
    if human_choice == "uncertain":
        return None
    return human_choice == evaluator_choice


def show_debug_metadata(item: dict[str, Any]) -> None:
    st.markdown("#### Hidden metadata")
    st.json({
        "item_id": item.get("item_id"),
        "sample_id": item.get("sample_id"),
        "pair_kind": item.get("pair_kind"),
        "pair_bucket": item.get("pair_bucket"),
        "model_pair": item.get("model_pair"),
        "evaluator_preference": item.get("evaluator_preference"),
        "evaluator_score_delta": item.get("evaluator_score_delta"),
        "side_a": {
            key: item.get("side_a", {}).get(key)
            for key in ["source_type", "model", "evaluator_score", "reason_prediction"]
        },
        "side_b": {
            key: item.get("side_b", {}).get(key)
            for key in ["source_type", "model", "evaluator_score", "reason_prediction"]
        },
        "gold_score": item.get("gold_score"),
        "gold_reason": item.get("gold_reason"),
    })


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title=args.page_title, layout="wide")
    st.title(args.page_title)

    if not os.path.exists(args.items_jsonl):
        st.error(f"Item file does not exist: {args.items_jsonl}")
        st.stop()

    items = load_items(args.items_jsonl)
    if not items:
        st.error("No annotation items were loaded.")
        st.stop()

    with st.sidebar:
        st.header("Settings")
        annotator_id = st.text_input("Annotator ID", value="annotator_01")
        if not safe_annotator_id(annotator_id):
            st.error("Annotator ID is required.")
            st.stop()

        out_path = annotation_path(args.output_dir, annotator_id)
        annotations = load_annotations(out_path)
        total_items = len(items)
        subset_labels = {
            key: subset_caption(key, *subset_bounds(total_items, key), total_items)
            for key in SUBSET_OPTIONS
        }
        subset_label = st.selectbox(
            "Annotation subset",
            options=list(subset_labels.values()),
            index=0,
        )
        subset_key = next(
            key for key, label in subset_labels.items() if label == subset_label
        )
        subset_start, subset_end = subset_bounds(total_items, subset_key)
        selected_items = items[subset_start:subset_end]
        total = len(selected_items)
        if total == 0:
            st.error("The selected annotation subset has no items.")
            st.stop()

        if st.session_state.get("loaded_subset_key") != subset_key:
            st.session_state.current_idx = first_unannotated_index(selected_items, annotations)
            st.session_state.loaded_subset_key = subset_key

        selected_item_ids = {str(item.get("item_id")) for item in selected_items}
        done = len(set(annotations) & selected_item_ids)
        st.progress(done / total if total else 0.0)
        st.write(f"Subset progress: **{done} / {total}**")
        st.caption(f"Full item set: {total_items} items")
        st.caption(f"Saving to `{out_path}`")

        if "current_idx" not in st.session_state:
            st.session_state.current_idx = first_unannotated_index(selected_items, annotations)
        if st.button("Go to next unlabeled item"):
            st.session_state.current_idx = first_unannotated_index(selected_items, annotations)
            st.rerun()

        max_idx = max(0, total - 1)
        chosen_idx = st.number_input(
            "Subset item index",
            min_value=0,
            max_value=max_idx,
            value=min(int(st.session_state.current_idx), max_idx),
            step=1,
        )
        st.session_state.current_idx = int(chosen_idx)
        absolute_item_index = subset_start + int(st.session_state.current_idx)
        st.caption(f"Current full-set item index: {absolute_item_index + 1} / {total_items}")

        show_profile = st.checkbox("Show user profile", value=True)
        show_preference_evidence = st.checkbox("Show preference evidence", value=True)
        show_debug = st.checkbox("Show hidden metadata", value=args.show_debug_default)
        st.caption("Model names and evaluator scores should remain hidden during normal annotation.")

    item = selected_items[int(st.session_state.current_idx)]
    item_id = str(item.get("item_id"))
    existing = annotations.get(item_id)

    header_cols = st.columns([2, 1, 1, 1])
    header_cols[0].markdown(f"**Item:** `{item_id}`")
    header_cols[1].markdown(f"**Task:** {item.get('target_task', '')}")
    header_cols[2].markdown(f"**User:** {item.get('user', '')}")
    header_cols[3].markdown(f"**Turn:** {item.get('turn_idx', '')}")
    if existing:
        st.info("This item already has an annotation. Submitting again will append a newer record and override it in later analysis.")

    with st.expander("Task context", expanded=True):
        st.write(item.get("task_context", ""))
    if show_profile:
        with st.expander("User profile", expanded=False):
            format_profile(item.get("profile", {}))
    if show_preference_evidence:
        with st.expander("User preference evidence", expanded=True):
            render_user_preference_evidence(item.get("user_preference_evidence", {}))

    left, right = st.columns([1.1, 1.2])
    with left:
        st.subheader("Conversation prefix")
        render_conversation(item.get("dialogue_prefix", []))
    with right:
        st.subheader("Current user request")
        st.write(item.get("current_user_request", ""))
        if show_debug:
            show_debug_metadata(item)

    resp_a, resp_b = st.columns(2)
    with resp_a:
        render_response("A", item.get("side_a", {}))
    with resp_b:
        render_response("B", item.get("side_b", {}))

    with st.form(key=f"pairwise_form_{item_id}"):
        st.subheader("Annotation")
        preference_label = st.radio(
            "Which response better satisfies the user in this turn?",
            options=list(PREFERENCE_OPTIONS),
            horizontal=True,
        )
        confidence = st.radio(
            "Confidence",
            options=CONFIDENCE_OPTIONS,
            index=1,
            horizontal=True,
        )
        reason_categories = st.multiselect(
            "Why? Select any applicable categories.",
            options=REASON_OPTIONS,
        )
        comment = st.text_area("Optional comment", height=90)
        submitted = st.form_submit_button("Submit and continue")

    if submitted:
        human_choice = PREFERENCE_OPTIONS[preference_label]
        annotation = {
            "annotator_id": safe_annotator_id(annotator_id),
            "annotation_subset": subset_key,
            "subset_item_index": int(st.session_state.current_idx),
            "source_item_index": subset_start + int(st.session_state.current_idx),
            "source_item_total": len(items),
            "item_id": item_id,
            "sample_id": item.get("sample_id"),
            "user": item.get("user"),
            "target_task": item.get("target_task"),
            "turn_idx": item.get("turn_idx"),
            "human_preference": human_choice,
            "preferred_response_id": human_choice_to_response_id(item, human_choice),
            "confidence": confidence,
            "reason_categories": reason_categories,
            "comment": comment.strip(),
            "evaluator_preference": item.get("evaluator_preference"),
            "agreement_with_evaluator": agreement_with_evaluator(item, human_choice),
            "side_a_response_id": item.get("side_a", {}).get("response_id"),
            "side_b_response_id": item.get("side_b", {}).get("response_id"),
            "pair_kind": item.get("pair_kind"),
            "pair_bucket": item.get("pair_bucket"),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        append_annotation(out_path, annotation)
        st.session_state.current_idx = min(
            first_unannotated_index(selected_items, load_annotations(out_path)),
            total - 1,
        )
        st.rerun()


if __name__ == "__main__":
    main()
