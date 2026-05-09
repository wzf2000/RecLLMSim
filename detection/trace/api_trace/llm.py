from __future__ import annotations

from openai.types.chat import ChatCompletionMessageParam
from tenacity import retry, stop_after_attempt, wait_fixed

from lib.llm import client
from .schema import ReflectionAnswer, TraceAnswer


@retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
def predict_with_parse(messages: list[ChatCompletionMessageParam], model: str) -> tuple[TraceAnswer, str, str]:
    response = client.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=0.6,
        response_format=TraceAnswer,
        timeout=60,
    ).choices[0].message

    if response.parsed:
        raw_content = response.content if response.content else ""
        reasoning_content = response.reasoning_content if hasattr(response, "reasoning_content") else ""  # type: ignore
        return response.parsed, reasoning_content or "", raw_content

    if response.refusal:
        raise RuntimeError(f"Refusal: {response.refusal}")
    raise RuntimeError("Parse failed: empty parsed content")




@retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
def reflect_with_parse(messages: list[ChatCompletionMessageParam], model: str) -> tuple[ReflectionAnswer, str, str]:
    response = client.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=0.6,
        response_format=ReflectionAnswer,
        timeout=60,
    ).choices[0].message

    if response.parsed:
        raw_content = response.content if response.content else ""
        reasoning_content = response.reasoning_content if hasattr(response, "reasoning_content") else ""  # type: ignore
        return response.parsed, reasoning_content or "", raw_content
    if response.refusal:
        raise RuntimeError(f"Refusal: {response.refusal}")
    raise RuntimeError("Parse failed: empty parsed content")


