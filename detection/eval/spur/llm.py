from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_fixed

from lib.llm import client


@retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
def _call_llm(messages: list[dict], model: str) -> str:
    response = client.chat.completions.create(
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
