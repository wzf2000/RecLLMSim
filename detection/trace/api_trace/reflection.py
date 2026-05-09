from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from loguru import logger
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)
from tqdm import tqdm

from .io import dump_jsonl, load_jsonl
from .llm import reflect_with_parse


def get_error_type(pred_score: int, gold_score: int, pred_reason: str, gold_reason: str) -> str:
    score_wrong = pred_score != gold_score
    reason_wrong = pred_reason != gold_reason
    if score_wrong and reason_wrong:
        return "both"
    if score_wrong:
        return "score"
    return "reason"


def build_reflection_user_feedback(record: dict, error_type: str) -> str:
    if error_type == "score":
        focus = (
            "本次主要错误是满意度分数判断。请重点反思你对用户情绪强度、需求满足程度、"
            "对话目标达成度的刻画是否偏高/偏低，原因标签可作为辅助证据。"
        )
    elif error_type == "reason":
        focus = (
            "本次主要错误是原因标签判断。请重点反思你如何区分相近原因类别，"
            "并指出导致你误判标签边界的关键语义线索。分数可作为辅助证据。"
        )
    else:
        focus = (
            "本次分数和原因均有偏差。请先分析两类错误的耦合关系，"
            "再给出联合修正后的推理。"
        )

    return (
        "你上一轮给出的预测与标注存在偏差。下面给出标准答案，请你基于正确标签进行反思：\n"
        f"- 错误类型: {error_type}\n"
        f'- 你预测的分数: {record.get("prediction")}\n'
        f'- 你预测的原因: {record.get("reason_prediction")}\n'
        f'- 正确分数: {record.get("gold_score")}\n'
        f'- 正确原因: {record.get("gold_reason")}\n\n'
        f"{focus}\n\n"
        "请重新审视原始上下文，重点说明你之前可能忽略或误判了哪些信号，"
        "然后给出基于正确分数/原因的重写推理过程。\n\n"
        "请严格输出 JSON：\n"
        "{\n"
        '  "problem_analysis": "你此前预测错误的关键原因分析",\n'
        '  "reflection": "你从这次错误中得到的反思",\n'
        '  "revised_reasoning": "基于正确分数和原因重写后的完整推理过程"\n'
        "}\n"
    )




def generate_reflections_from_file(
    model: str,
    input_jsonl: str,
    reflection_output_jsonl: str,
    max_workers: int = 8,
) -> None:
    records = load_jsonl(input_jsonl)
    if not records:
        raise ValueError(f"Input file is empty or not found: {input_jsonl}")

    output_lock = Lock()
    results: list[dict | None] = [None for _ in range(len(records))]

    def process_one(i: int, record: dict) -> tuple[int, dict]:
        pred_score = int(record.get("prediction", -1))
        gold_score = int(record.get("gold_score", -2))
        pred_reason = str(record.get("reason_prediction", ""))
        gold_reason = str(record.get("gold_reason", ""))
        is_wrong = pred_score != gold_score or pred_reason != gold_reason
        if not is_wrong:
            return i, record
        error_type = get_error_type(pred_score, gold_score, pred_reason, gold_reason)

        base_messages = record.get("messages", [])
        if not isinstance(base_messages, list):
            base_messages = []
        reflection_messages: list[ChatCompletionMessageParam] = list(base_messages)

        assistant_content = record.get("raw_content")
        if not isinstance(assistant_content, str) or not assistant_content.strip():
            # 兼容旧版轨迹字段
            assistant_content = record.get("raw_output")
        if isinstance(assistant_content, str) and assistant_content.strip():
            reflection_messages.append(
                ChatCompletionAssistantMessageParam(role="assistant", content=assistant_content)
            )
        reflection_messages.append(
            ChatCompletionUserMessageParam(
                role="user",
                content=build_reflection_user_feedback(record, error_type),
            )
        )

        reflection_answer, reflection_reasoning_content, reflection_raw_content = reflect_with_parse(reflection_messages, model)
        new_record = dict(record)
        new_record["reflection"] = {
            "error_type": error_type,
            "problem_analysis": reflection_answer.problem_analysis,
            "reflection": reflection_answer.reflection,
            "revised_reasoning": reflection_answer.revised_reasoning,
            "reasoning_content": reflection_reasoning_content,
            "raw_content": reflection_raw_content,
        }
        return i, new_record

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_one, i, record): i
            for i, record in enumerate(records)
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx = futures[future]
            try:
                i, new_record = future.result()
                with output_lock:
                    results[i] = new_record
            except Exception as e:
                logger.error(f"Reflection for record {idx} failed: {e}")
                with output_lock:
                    results[idx] = records[idx]

    final_rows = [row for row in results if row is not None]
    dump_jsonl(reflection_output_jsonl, final_rows)
    logger.info(f"Reflection finished. Total: {len(final_rows)}, output: {reflection_output_jsonl}")


