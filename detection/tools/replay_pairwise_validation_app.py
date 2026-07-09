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
        total = len(items)
        done = len(set(annotations) & {str(item.get("item_id")) for item in items})
        st.progress(done / total if total else 0.0)
        st.write(f"Progress: **{done} / {total}**")
        st.caption(f"Saving to `{out_path}`")

        if "current_idx" not in st.session_state:
            st.session_state.current_idx = first_unannotated_index(items, annotations)
        if st.button("Go to next unlabeled item"):
            st.session_state.current_idx = first_unannotated_index(items, annotations)
            st.rerun()

        max_idx = max(0, total - 1)
        chosen_idx = st.number_input(
            "Item index",
            min_value=0,
            max_value=max_idx,
            value=min(int(st.session_state.current_idx), max_idx),
            step=1,
        )
        st.session_state.current_idx = int(chosen_idx)

        show_profile = st.checkbox("Show user profile", value=True)
        show_debug = st.checkbox("Show hidden metadata", value=args.show_debug_default)
        st.caption("Model names and evaluator scores should remain hidden during normal annotation.")

    item = items[int(st.session_state.current_idx)]
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
            first_unannotated_index(items, load_annotations(out_path)),
            total - 1,
        )
        st.rerun()


if __name__ == "__main__":
    main()
