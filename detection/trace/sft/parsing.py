from __future__ import annotations

import json
import re


# Qwen3 chat_template：思考在 <think>...</think> 内，其后为可见正文（与 HF 官方 jinja 一致）
QWEN3_THINK_BEGIN = "\u003cthink\u003e"
QWEN3_THINK_END = "\u003c/think\u003e"
# 部分 API / 旧脚本用反引号包裹的 think 标记（见 detection/llm.py）
_BACKTICK_THINK_END = "\u0060think\u0060"

_THINK_VISIBLE_SPLIT_MARKERS: tuple[str, ...] = (
    QWEN3_THINK_END,
    _BACKTICK_THINK_END,
)


def strip_visible_answer_after_think(text: str) -> str:
    """去掉思考段，仅保留其后对用户可见的续写（通常为 JSON 或 reasoning+JSON）。"""
    s = text.strip()
    best_cut = -1
    best_end = 0
    for m in _THINK_VISIBLE_SPLIT_MARKERS:
        j = s.rfind(m)
        if j > best_cut:
            best_cut = j
            best_end = j + len(m)
    if best_cut >= 0:
        return s[best_end:].lstrip()
    return s


def _unwrap_outer_code_fence(s: str) -> str:
    t = s.strip()
    if not t.startswith("```"):
        return t
    lines = t.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _segment_looks_like_trace_json(seg: str) -> bool:
    t = seg.strip()
    if not t:
        return False
    if t.startswith("```"):
        return True
    return '"classification"' in t and '"reason"' in t


def _try_parse_trace_answer_obj(json_str: str) -> tuple[int, str] | None:
    try:
        obj = json.loads(json_str)
        if not isinstance(obj, dict):
            return None
        c = int(obj["classification"])
        r = str(obj.get("reason", "")).strip()
        if not (1 <= c <= 5):
            return None
        return c, r
    except Exception:
        pass
    # 生成被截断时（analysis 字段过长），尝试 regex 提取 classification 和 reason
    m_c = re.search(r'"classification"\s*:\s*([1-5])', json_str)
    m_r = re.search(r'"reason"\s*:\s*"([^"]*)"', json_str)
    if m_c is not None:
        c = int(m_c.group(1))
        r = m_r.group(1).strip() if m_r is not None else ""
        return c, r
    return None


def _balanced_json_spans(s: str) -> list[tuple[int, str]]:
    """枚举 s 中每个从 '{' 开始的平衡括号子串 (start_index, substring)。"""
    spans: list[tuple[int, str]] = []
    n = len(s)
    i = 0
    while i < n:
        if s[i] != "{":
            i += 1
            continue
        start = i
        depth = 0
        j = i
        while j < n:
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    spans.append((start, s[start : j + 1]))
                    break
            j += 1
        i += 1
    return spans


def parse_model_json(text: str) -> tuple[int | None, str | None]:
    """
    从模型生成文本中解析满意度 JSON（classification + reason）。

    适配：1) 前置长推理 + 末尾 JSON（与 include_reasoning_content 训练目标一致）；
    2) 推理中含 '{' 时不能误用首段括号；
    3) 先去掉 Qwen3 的 /think 闭合标记（及反引号 think 变体）之后的可见段，再解析 JSON。
    """
    s = strip_visible_answer_after_think(text)
    s = _unwrap_outer_code_fence(s)

    # 与 migrate_trace_jsonl_format 一致：从末段双换行块中找 JSON
    parts = s.split("\n\n")
    for k in range(len(parts) - 1, -1, -1):
        tail = "\n\n".join(parts[k:]).strip()
        if not _segment_looks_like_trace_json(tail):
            continue
        tail_u = _unwrap_outer_code_fence(tail)
        got = _try_parse_trace_answer_obj(tail_u)
        if got is not None:
            return got[0], got[1]

    # 扫描所有平衡 {...}，取「起始位置最靠后」且能解析为 TraceAnswer 的一段
    best: tuple[int, int, str] | None = None  # (start_idx, classification, reason)
    for start, chunk in _balanced_json_spans(s):
        got = _try_parse_trace_answer_obj(chunk)
        if got is None:
            continue
        if best is None or start >= best[0]:
            best = (start, got[0], got[1])
    if best is not None:
        return best[1], best[2]

    # 末段再尝试整段是否为 JSON（无换行分隔时）
    got = _try_parse_trace_answer_obj(s.strip())
    if got is not None:
        return got[0], got[1]

    for ch in s:
        if ch in "12345":
            return int(ch), None
    return None, None
