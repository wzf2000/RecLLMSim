from __future__ import annotations

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed

from lib.llm import client

_client = client


def configure_client(base_url: str = "", api_key: str = "") -> None:
    """Override the default project client, e.g. for local vLLM endpoints."""
    global _client
    if base_url:
        _client = OpenAI(
            base_url=base_url,
            api_key=api_key or "EMPTY",
        )


@retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
def _call_llm(messages: list[dict], model: str) -> str:
    response = _client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        timeout=90,
    ).choices[0].message
    content = (response.content or "").strip()
    if content.startswith("<think>"):
        content = content.split("</think>", 1)[1].strip()
    if content.startswith("```json") and content.endswith("```"):
        content = content[7:-3].strip()
    return content


def _sys_user(system: str, user: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
