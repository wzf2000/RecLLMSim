"""Streamlit app for pairwise human validation of replayed responses."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

import streamlit as st


DEFAULT_PAGE_TITLE = "Replay Pairwise Validation"

LANGUAGE_OPTIONS = {
    "中文": "zh",
    "English": "en",
}

TEXT = {
    "app_title": {
        "zh": "Replay 成对人工标注",
        "en": "Replay Pairwise Validation",
    },
    "settings": {"zh": "设置", "en": "Settings"},
    "language": {"zh": "语言 / Language", "en": "Language / 语言"},
    "annotator_id": {"zh": "标注者 ID", "en": "Annotator ID"},
    "annotator_required": {"zh": "请填写标注者 ID。", "en": "Annotator ID is required."},
    "annotation_subset": {"zh": "标注范围", "en": "Annotation subset"},
    "all_items": {"zh": "全部样本", "en": "All items"},
    "first_half": {"zh": "前半部分", "en": "First half"},
    "second_half": {"zh": "后半部分", "en": "Second half"},
    "no_items": {"zh": "没有样本", "en": "No items"},
    "subset_empty": {"zh": "当前选择的标注范围没有样本。", "en": "The selected annotation subset has no items."},
    "subset_progress": {"zh": "当前范围进度", "en": "Subset progress"},
    "full_item_set": {"zh": "完整样本集", "en": "Full item set"},
    "saving_to": {"zh": "保存到", "en": "Saving to"},
    "next_unlabeled": {"zh": "跳转到下一个未标注样本", "en": "Go to next unlabeled item"},
    "subset_item_index": {"zh": "当前范围内的样本编号", "en": "Subset item index"},
    "current_full_index": {"zh": "完整样本集中的当前位置", "en": "Current full-set item index"},
    "show_profile": {"zh": "显示用户画像", "en": "Show user profile"},
    "show_preference_evidence": {"zh": "显示用户偏好证据", "en": "Show preference evidence"},
    "show_debug": {"zh": "显示隐藏调试信息", "en": "Show hidden metadata"},
    "hidden_hint": {
        "zh": "正常标注时应隐藏模型名称和 evaluator 分数。",
        "en": "Model names and evaluator scores should remain hidden during normal annotation.",
    },
    "item_file_missing": {"zh": "样本文件不存在", "en": "Item file does not exist"},
    "no_loaded_items": {"zh": "没有加载到标注样本。", "en": "No annotation items were loaded."},
    "item": {"zh": "样本", "en": "Item"},
    "task": {"zh": "任务", "en": "Task"},
    "user": {"zh": "用户", "en": "User"},
    "turn": {"zh": "轮次", "en": "Turn"},
    "already_annotated": {
        "zh": "该样本已经有标注记录。再次提交会追加一条新记录，后续分析会以最新记录为准。",
        "en": "This item already has an annotation. Submitting again will append a newer record and override it in later analysis.",
    },
    "task_context": {"zh": "任务背景", "en": "Task context"},
    "user_profile": {"zh": "用户画像", "en": "User profile"},
    "preference_evidence": {"zh": "用户偏好证据", "en": "User preference evidence"},
    "conversation_prefix": {"zh": "当前对话历史", "en": "Conversation prefix"},
    "current_user_request": {"zh": "当前用户请求", "en": "Current user request"},
    "response": {"zh": "回复", "en": "Response"},
    "annotation": {"zh": "标注", "en": "Annotation"},
    "preference_question": {
        "zh": "在本轮中，哪一个回复更能让该用户满意？",
        "en": "Which response better satisfies the user in this turn?",
    },
    "confidence": {"zh": "置信度", "en": "Confidence"},
    "reason_question": {"zh": "原因：可多选", "en": "Why? Select any applicable categories."},
    "comment": {"zh": "可选备注", "en": "Optional comment"},
    "submit": {"zh": "提交并继续", "en": "Submit and continue"},
    "no_profile": {"zh": "该样本没有可用的用户画像。", "en": "No profile is available for this item."},
    "no_score_dist": {
        "zh": "没有可用的历史评分分布。",
        "en": "No historical score distribution is available.",
    },
    "historical_mean": {"zh": "历史平均分", "en": "Historical mean"},
    "source_turns": {"zh": "个历史助手轮次", "en": "source-history assistant turns"},
    "score": {"zh": "分数", "en": "Score"},
    "count": {"zh": "数量", "en": "Count"},
    "score_caption": {
        "zh": "SAT = 4-5 分；Neutral = 3 分；低分 = 1-2 分。",
        "en": "SAT = scores 4-5; Neutral = score 3; low = scores 1-2.",
    },
    "sat_rate": {"zh": "SAT 比例", "en": "SAT rate"},
    "neutral_rate": {"zh": "Neutral 比例", "en": "Neutral rate"},
    "low_rate": {"zh": "1/2 低分比例", "en": "low-score rate"},
    "historical_reason": {"zh": "历史标注原因", "en": "Historical reason"},
    "historical_user_request": {"zh": "历史用户请求", "en": "Historical user request"},
    "historical_response": {"zh": "历史助手回复", "en": "Historical assistant response"},
    "no_preference_evidence": {
        "zh": "该样本没有可用的历史偏好证据。",
        "en": "No source-history preference evidence is available for this item.",
    },
    "evidence_caption": {
        "zh": "这些信息来自该用户在其他场景中的带标注历史对话，不包含当前隐藏 evaluator 分数或模型身份。",
        "en": "Generated from this user's labeled source-history conversations in other scenarios. It does not include the current hidden evaluator score or model identity.",
    },
    "preference_summary": {"zh": "偏好摘要", "en": "Preference summary"},
    "low_side_reasons": {"zh": "常见低分 / 中立原因", "en": "Common low-side / neutral reasons"},
    "reason": {"zh": "原因", "en": "Reason"},
    "anchor_examples": {"zh": "历史参考样例", "en": "Historical anchor examples"},
    "high_examples": {"zh": "高分样例", "en": "High-score examples"},
    "low_examples": {"zh": "低分 / 中立样例", "en": "Low / neutral examples"},
    "no_high_anchors": {"zh": "没有高分参考样例。", "en": "No high-score anchors."},
    "no_low_anchors": {"zh": "没有低分/中立参考样例。", "en": "No low/neutral anchors."},
    "no_prefix": {"zh": "没有可用的对话历史。", "en": "No conversation prefix is available."},
    "empty_response": {"zh": "_空回复。_", "en": "_Empty response._"},
    "hidden_metadata": {"zh": "隐藏元数据", "en": "Hidden metadata"},
}

PREFERENCE_OPTIONS = ["a", "b", "tie", "uncertain"]
PREFERENCE_LABELS = {
    "a": {"zh": "A 更好", "en": "A is better"},
    "b": {"zh": "B 更好", "en": "B is better"},
    "tie": {"zh": "两者相近 / 难分高下", "en": "Tie / similar"},
    "uncertain": {"zh": "无法判断", "en": "Cannot judge"},
}

CONFIDENCE_OPTIONS = ["high", "medium", "low"]
CONFIDENCE_LABELS = {
    "high": {"zh": "高", "en": "high"},
    "medium": {"zh": "中", "en": "medium"},
    "low": {"zh": "低", "en": "low"},
}

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
REASON_LABELS = {
    "better addresses the user request": {"zh": "更好地回应当前用户请求", "en": "better addresses the user request"},
    "more personalized": {"zh": "更符合该用户的个人偏好", "en": "more personalized"},
    "more specific or actionable": {"zh": "更具体 / 更可执行", "en": "more specific or actionable"},
    "better follows constraints": {"zh": "更好地遵守限制条件", "en": "better follows constraints"},
    "better structure or readability": {"zh": "结构或可读性更好", "en": "better structure or readability"},
    "more reliable or safer": {"zh": "更可靠 / 更安全", "en": "more reliable or safer"},
    "less verbose": {"zh": "更简洁、不啰嗦", "en": "less verbose"},
    "other": {"zh": "其他", "en": "other"},
}

SUBSET_OPTIONS = {
    "all": "all_items",
    "first_half": "first_half",
    "second_half": "second_half",
}

PROFILE_FIELD_LABELS = [
    ("gender", {"zh": "性别", "en": "Gender"}),
    ("age", {"zh": "年龄", "en": "Age"}),
    ("occupation", {"zh": "职业", "en": "Occupation"}),
    ("background", {"zh": "背景", "en": "Background"}),
    ("personality", {"zh": "性格", "en": "Personality"}),
    ("daily_interests", {"zh": "日常兴趣", "en": "Daily interests"}),
    ("travel_habits", {"zh": "旅行习惯", "en": "Travel habits"}),
    ("dining_preferences", {"zh": "饮食偏好", "en": "Dining preferences"}),
    ("spending_habits", {"zh": "消费习惯", "en": "Spending habits"}),
    ("other_aspects", {"zh": "其他方面", "en": "Other aspects"}),
]


def t(key: str, lang: str) -> str:
    return TEXT[key][lang]


def localized(mapping: dict[str, dict[str, str]], key: str, lang: str) -> str:
    return mapping.get(key, {}).get(lang, str(key))


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
    parser.add_argument("--page_title", default=DEFAULT_PAGE_TITLE)
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


def subset_caption(subset_key: str, start: int, end: int, total: int, lang: str) -> str:
    if total <= 0:
        return t("no_items", lang)
    if subset_key == "all":
        return f"{t('all_items', lang)}: 1-{total}"
    return f"{t(SUBSET_OPTIONS[subset_key], lang)}: {start + 1}-{end} / {total}"


def format_profile(profile: dict[str, Any], lang: str) -> None:
    if not profile:
        st.caption(t("no_profile", lang))
        return
    for key, labels in PROFILE_FIELD_LABELS:
        value = profile.get(key)
        if value in (None, "", []):
            continue
        if isinstance(value, list):
            value_text = ", ".join(str(v) for v in value)
        else:
            value_text = str(value)
        st.markdown(f"**{labels[lang]}:** {value_text}")


def render_score_distribution(dist: dict[str, Any], lang: str) -> None:
    if not dist or not dist.get("total_turns"):
        st.caption(t("no_score_dist", lang))
        return
    counts = dist.get("counts", {})
    rows = [
        {
            t("score", lang): score,
            t("count", lang): int(counts.get(str(score), 0)),
        }
        for score in range(1, 6)
    ]
    st.markdown(
        f"**{t('historical_mean', lang)}:** {float(dist.get('mean', 0.0)):.2f} "
        f"({int(dist.get('total_turns', 0))} {t('source_turns', lang)})"
    )
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.caption(
        f"{t('score_caption', lang)} "
        f"{t('sat_rate', lang)} {float(dist.get('sat_rate') or 0.0):.1%}; "
        f"{t('neutral_rate', lang)} {float(dist.get('neutral_rate') or 0.0):.1%}; "
        f"{t('low_rate', lang)} {float(dist.get('low_rate') or 0.0):.1%}."
    )


def render_anchor_example(example: dict[str, Any], idx: int, lang: str) -> None:
    title = (
        f"{idx}. {example.get('source_task', '')} / "
        f"{example.get('source_file', '')} / {t('turn', lang)} {example.get('turn_idx', '')} "
        f"/ {t('score', lang)} {example.get('score', '')}"
    )
    with st.expander(title, expanded=False):
        reason = example.get("reason")
        if reason:
            st.markdown(f"**{t('historical_reason', lang)}:** {reason}")
        st.markdown(f"**{t('historical_user_request', lang)}**")
        st.write(example.get("user_request", ""))
        st.markdown(f"**{t('historical_response', lang)}**")
        st.write(example.get("assistant_response", ""))


def zh_preference_summary(evidence: dict[str, Any]) -> list[str]:
    dist = evidence.get("score_distribution", {})
    tasks = evidence.get("source_history_tasks", [])
    total = int(dist.get("total_turns") or 0)
    if total <= 0:
        return ["该用户在其他场景中没有可用的历史满意度证据。"]
    mean = float(dist.get("mean") or 0.0)
    sat_rate = float(dist.get("sat_rate") or 0.0)
    neutral_rate = float(dist.get("neutral_rate") or 0.0)
    low_rate = float(dist.get("low_rate") or 0.0)
    summary = [
        f"历史证据覆盖该用户在 {len(tasks)} 个其他场景中的 {total} 个助手回复轮次：{', '.join(tasks)}。",
        f"历史平均分为 {mean:.2f}；SAT 比例 {sat_rate:.1%}，Neutral 比例 {neutral_rate:.1%}，1/2 低分比例 {low_rate:.1%}。",
    ]
    if mean >= 4.25 and low_rate <= 0.08:
        summary.append("该用户在历史中整体较容易满意，但仍需要检查回复是否满足当前具体约束。")
    elif mean <= 3.85:
        summary.append("该用户在历史中相对严格，泛泛而谈或不够具体的回复更可能不被偏好。")
    else:
        summary.append("该用户历史评分较混合，建议结合当前请求和下方历史参考样例判断。")
    reasons = evidence.get("low_side_reasons", [])
    if reasons:
        reason_text = "，".join(
            f"{item.get('reason', '')}（{int(item.get('count', 0))}）"
            for item in reasons[:3]
        )
        summary.append(f"历史中常见低分/中立原因包括：{reason_text}。")
    summary.append("该摘要由历史标注统计和样例模板生成，不是 evaluator memory 的原文。")
    return summary


def render_user_preference_evidence(evidence: dict[str, Any], lang: str) -> None:
    if not evidence:
        st.caption(t("no_preference_evidence", lang))
        return

    st.caption(t("evidence_caption", lang))
    summary = zh_preference_summary(evidence) if lang == "zh" else evidence.get("summary", [])
    if summary:
        st.markdown(f"**{t('preference_summary', lang)}**")
        for bullet in summary:
            st.markdown(f"- {bullet}")

    render_score_distribution(evidence.get("score_distribution", {}), lang)

    reasons = evidence.get("low_side_reasons", [])
    if reasons:
        st.markdown(f"**{t('low_side_reasons', lang)}**")
        reason_rows = [
            {t("reason", lang): item.get("reason", ""), t("count", lang): int(item.get("count", 0))}
            for item in reasons
        ]
        st.dataframe(reason_rows, hide_index=True, use_container_width=True)

    anchors = evidence.get("anchor_examples", {})
    high_examples = anchors.get("high_score", [])
    low_examples = anchors.get("low_or_neutral", [])
    if high_examples or low_examples:
        st.markdown(f"**{t('anchor_examples', lang)}**")
        high_col, low_col = st.columns(2)
        with high_col:
            st.markdown(t("high_examples", lang))
            if not high_examples:
                st.caption(t("no_high_anchors", lang))
            for idx, example in enumerate(high_examples, 1):
                render_anchor_example(example, idx, lang)
        with low_col:
            st.markdown(t("low_examples", lang))
            if not low_examples:
                st.caption(t("no_low_anchors", lang))
            for idx, example in enumerate(low_examples, 1):
                render_anchor_example(example, idx, lang)


def render_conversation(prefix: list[dict[str, Any]], lang: str) -> None:
    if not prefix:
        st.caption(t("no_prefix", lang))
        return
    with st.container(height=460):
        for utt in prefix:
            role = "user" if utt.get("role") == "user" else "assistant"
            with st.chat_message(role):
                st.write(utt.get("content", ""))


def render_response(label: str, side: dict[str, Any], lang: str) -> None:
    st.subheader(f"{t('response', lang)} {label}")
    with st.container(height=460):
        st.markdown(side.get("response", "") or t("empty_response", lang))


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


def show_debug_metadata(item: dict[str, Any], lang: str) -> None:
    st.markdown(f"#### {t('hidden_metadata', lang)}")
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

    if not os.path.exists(args.items_jsonl):
        st.error(f"{t('item_file_missing', 'zh')} / {t('item_file_missing', 'en')}: {args.items_jsonl}")
        st.stop()

    items = load_items(args.items_jsonl)
    if not items:
        st.error(f"{t('no_loaded_items', 'zh')} / {t('no_loaded_items', 'en')}")
        st.stop()

    with st.sidebar:
        lang_label = st.selectbox(
            "语言 / Language",
            options=list(LANGUAGE_OPTIONS),
            index=0,
        )
        lang = LANGUAGE_OPTIONS[lang_label]

    st.title(t("app_title", lang))

    with st.sidebar:
        st.header(t("settings", lang))
        annotator_id = st.text_input(t("annotator_id", lang), value="annotator_01")
        if not safe_annotator_id(annotator_id):
            st.error(t("annotator_required", lang))
            st.stop()

        out_path = annotation_path(args.output_dir, annotator_id)
        annotations = load_annotations(out_path)
        total_items = len(items)
        subset_labels = {
            key: subset_caption(key, *subset_bounds(total_items, key), total_items, lang)
            for key in SUBSET_OPTIONS
        }
        subset_label = st.selectbox(
            t("annotation_subset", lang),
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
            st.error(t("subset_empty", lang))
            st.stop()

        if st.session_state.get("loaded_subset_key") != subset_key:
            st.session_state.current_idx = first_unannotated_index(selected_items, annotations)
            st.session_state.loaded_subset_key = subset_key

        selected_item_ids = {str(item.get("item_id")) for item in selected_items}
        done = len(set(annotations) & selected_item_ids)
        st.progress(done / total if total else 0.0)
        st.write(f"{t('subset_progress', lang)}: **{done} / {total}**")
        st.caption(f"{t('full_item_set', lang)}: {total_items}")
        st.caption(f"{t('saving_to', lang)} `{out_path}`")

        if "current_idx" not in st.session_state:
            st.session_state.current_idx = first_unannotated_index(selected_items, annotations)
        if st.button(t("next_unlabeled", lang)):
            st.session_state.current_idx = first_unannotated_index(selected_items, annotations)
            st.rerun()

        max_idx = max(0, total - 1)
        chosen_idx = st.number_input(
            t("subset_item_index", lang),
            min_value=0,
            max_value=max_idx,
            value=min(int(st.session_state.current_idx), max_idx),
            step=1,
        )
        st.session_state.current_idx = int(chosen_idx)
        absolute_item_index = subset_start + int(st.session_state.current_idx)
        st.caption(f"{t('current_full_index', lang)}: {absolute_item_index + 1} / {total_items}")

        show_profile = st.checkbox(t("show_profile", lang), value=True)
        show_preference_evidence = st.checkbox(t("show_preference_evidence", lang), value=True)
        show_debug = st.checkbox(t("show_debug", lang), value=args.show_debug_default)
        st.caption(t("hidden_hint", lang))

    item = selected_items[int(st.session_state.current_idx)]
    item_id = str(item.get("item_id"))
    existing = annotations.get(item_id)

    header_cols = st.columns([2, 1, 1, 1])
    header_cols[0].markdown(f"**{t('item', lang)}:** `{item_id}`")
    header_cols[1].markdown(f"**{t('task', lang)}:** {item.get('target_task', '')}")
    header_cols[2].markdown(f"**{t('user', lang)}:** {item.get('user', '')}")
    header_cols[3].markdown(f"**{t('turn', lang)}:** {item.get('turn_idx', '')}")
    if existing:
        st.info(t("already_annotated", lang))

    with st.expander(t("task_context", lang), expanded=True):
        st.write(item.get("task_context", ""))
    if show_profile:
        with st.expander(t("user_profile", lang), expanded=False):
            format_profile(item.get("profile", {}), lang)
    if show_preference_evidence:
        with st.expander(t("preference_evidence", lang), expanded=True):
            render_user_preference_evidence(item.get("user_preference_evidence", {}), lang)

    left, right = st.columns([1.1, 1.2])
    with left:
        st.subheader(t("conversation_prefix", lang))
        render_conversation(item.get("dialogue_prefix", []), lang)
    with right:
        st.subheader(t("current_user_request", lang))
        st.write(item.get("current_user_request", ""))
        if show_debug:
            show_debug_metadata(item, lang)

    resp_a, resp_b = st.columns(2)
    with resp_a:
        render_response("A", item.get("side_a", {}), lang)
    with resp_b:
        render_response("B", item.get("side_b", {}), lang)

    with st.form(key=f"pairwise_form_{item_id}"):
        st.subheader(t("annotation", lang))
        human_choice = st.radio(
            t("preference_question", lang),
            options=PREFERENCE_OPTIONS,
            format_func=lambda option: localized(PREFERENCE_LABELS, option, lang),
            horizontal=True,
        )
        confidence = st.radio(
            t("confidence", lang),
            options=CONFIDENCE_OPTIONS,
            format_func=lambda option: localized(CONFIDENCE_LABELS, option, lang),
            index=1,
            horizontal=True,
        )
        reason_categories = st.multiselect(
            t("reason_question", lang),
            options=REASON_OPTIONS,
            format_func=lambda option: localized(REASON_LABELS, option, lang),
        )
        comment = st.text_area(t("comment", lang), height=90)
        submitted = st.form_submit_button(t("submit", lang))

    if submitted:
        annotation = {
            "annotator_id": safe_annotator_id(annotator_id),
            "ui_language": lang,
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
