"""Structured-output helpers shared by trace collection scripts."""

from __future__ import annotations

import json
import re
from typing import TypeVar

from loguru import logger
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    def __init__(self, message: str, raw_text: str = "") -> None:
        super().__init__(message)
        self.raw_text = raw_text


def message_content_to_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif hasattr(item, "text"):
                parts.append(str(item.text))
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                parts.append(str(text))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def strip_generation_wrappers(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def normalize_quotes(text: str) -> str:
    return (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )


def coerce_prediction_payload(payload: dict) -> dict:
    coerced = dict(payload)
    cls = coerced.get("classification")
    if isinstance(cls, str):
        cls = cls.strip()
        if cls in {"1", "2", "3", "4", "5"}:
            coerced["classification"] = int(cls)
    needs_review = coerced.get("needs_refute_review")
    if isinstance(needs_review, str):
        lowered = needs_review.strip().lower()
        if lowered in {"true", "false"}:
            coerced["needs_refute_review"] = lowered == "true"
    passes_boundary = coerced.get("passes_satisfaction_boundary")
    if isinstance(passes_boundary, str):
        lowered = passes_boundary.strip().lower()
        if lowered in {"true", "false"}:
            coerced["passes_satisfaction_boundary"] = lowered == "true"
    boundary_score = coerced.get("boundary_score")
    if isinstance(boundary_score, str):
        stripped = boundary_score.strip()
        if stripped in {"3", "4"}:
            coerced["boundary_score"] = int(stripped)
    delta_score = coerced.get("delta_score")
    if isinstance(delta_score, str):
        stripped = delta_score.strip()
        if stripped in {"-2", "-1", "0", "1", "2"}:
            coerced["delta_score"] = int(stripped)
    prior_score = coerced.get("history_prior_score")
    if isinstance(prior_score, str):
        try:
            coerced["history_prior_score"] = float(prior_score.strip())
        except ValueError:
            pass
    for bool_key in ("strong_failure_evidence", "strong_excellence_evidence"):
        value = coerced.get(bool_key)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "false"}:
                coerced[bool_key] = lowered == "true"
    return coerced


def recover_structured_output(raw_text: str, response_model: type[T]) -> T | None:
    cleaned = normalize_quotes(strip_generation_wrappers(raw_text))
    if not cleaned:
        return None

    # Fast path: extract the outermost JSON object and validate directly.
    left = cleaned.find("{")
    right = cleaned.rfind("}")
    if left != -1 and right != -1 and right > left:
        candidate = cleaned[left:right + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return response_model.model_validate(coerce_prediction_payload(parsed))
            return response_model.model_validate(parsed)
        except Exception:
            pass
    else:
        candidate = cleaned

    # Fallback: tolerant field extraction for slightly malformed JSON.
    cls_match = re.search(r'"classification"\s*:\s*"?(?P<cls>[1-5])"?', candidate)
    reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', candidate)
    analysis_match = re.search(r'"analysis"\s*:\s*"([\s\S]*?)"\s*}', candidate)
    review_match = re.search(
        r'"needs_refute_review"\s*:\s*"?(true|false)"?',
        candidate,
        flags=re.IGNORECASE,
    )

    analysis = ""
    if analysis_match:
        analysis = analysis_match.group(1).strip()
    else:
        analysis_key = '"analysis"'
        idx = candidate.find(analysis_key)
        if idx != -1:
            tail = candidate[idx + len(analysis_key):]
            colon = tail.find(":")
            if colon != -1:
                value = tail[colon + 1:].strip()
                if value.startswith('"'):
                    value = value[1:]
                value = value.replace("</think>", "").replace("<think>", "").strip()
                if value.endswith("}"):
                    value = value[:-1].rstrip()
                if value.endswith('"'):
                    value = value[:-1]
                analysis = value.strip()

    if cls_match and reason_match and analysis:
        try:
            payload = {
                "classification": int(cls_match.group("cls")),
                "reason": reason_match.group(1).strip(),
                "analysis": analysis,
            }
            if review_match:
                payload["needs_refute_review"] = review_match.group(1).lower() == "true"
            return response_model.model_validate(payload)
        except Exception:
            return None

    return None


def structured_parse(
    client,
    prompt: str,
    model: str,
    response_model: type[T],
    temperature: float = 0.3,
    timeout: int = 120,
    system_msg: str = "You are an expert user behavior analyst.",
) -> T:
    """
    Unified structured-output call.

    OpenAI API and vLLM >= 0.6 both support json_schema response_format, and
    the OpenAI SDK .parse() path behaves consistently for the current callers.
    """
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=temperature,
        response_format=response_model,
        timeout=timeout,
    ).choices[0].message
    if response.parsed:
        return response.parsed

    raw_text = message_content_to_text(getattr(response, "content", ""))
    recovered = recover_structured_output(raw_text, response_model)
    if recovered is not None:
        logger.warning(
            f"Recovered malformed structured output for {response_model.__name__} "
            f"(model={model}, temp={temperature}, raw_len={len(raw_text)})"
        )
        return recovered

    preview = raw_text[:300].replace("\n", "\\n")
    raise StructuredOutputError(
        f"Structured parse failed: {response.refusal or 'no content'}; "
        f"raw_preview={preview}",
        raw_text=raw_text,
    )


def structured_parse_from_raw_text(
    client,
    prompt: str,
    model: str,
    response_model: type[T],
    temperature: float = 0.3,
    timeout: int = 120,
    system_msg: str = "You are an expert user behavior analyst.",
) -> T:
    """
    Raw text route:
    - Do not use SDK .parse().
    - Read message.content directly.
    - Strip wrappers, recover tolerant JSON, then validate against the schema.
    """
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        timeout=timeout,
    ).choices[0].message

    raw_text = message_content_to_text(getattr(response, "content", ""))
    recovered = recover_structured_output(raw_text, response_model)
    if recovered is not None:
        return recovered

    preview = raw_text[:300].replace("\n", "\\n")
    raise StructuredOutputError(
        f"Raw structured parse failed: {response.refusal or 'no content'}; "
        f"raw_preview={preview}",
        raw_text=raw_text,
    )


# Backward-compatible aliases for older internal imports/tests.
_message_content_to_text = message_content_to_text
_strip_generation_wrappers = strip_generation_wrappers
_normalize_quotes = normalize_quotes
_coerce_prediction_payload = coerce_prediction_payload
_recover_structured_output = recover_structured_output
