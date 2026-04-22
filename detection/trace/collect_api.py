import json
import os
import random
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from loguru import logger
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

from lib.llm import client
from lib.data_split import split_by_user_group_shuffle_split
from lib.metric_statistics import get_satisfaction_data
from lib.satisfaction_constants import (
    get_dissatisfied_reasons,
    get_reason_to_id,
    is_reason_valid_for_score,
    normalize_reason_for_score,
)
from predictor.bert import format_profile


class TraceAnswer(BaseModel):
    classification: int = Field(ge=1, le=5)
    reason: str
    analysis: str


class ReflectionAnswer(BaseModel):
    problem_analysis: str
    reflection: str
    revised_reasoning: str


def preprocess_to_rows(data_list: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for sample in data_list:
        persona = format_profile(sample["profile"])
        task_context = sample["task_context"]
        history_window: list[str] = []
        assistant_turn_idx = 0

        for utt in sample["history"]:
            if utt["role"] == "assistant":
                rows.append(
                    {
                        "persona": persona,
                        "task_context": task_context,
                        "history": "\n".join(history_window),
                        "assistant_reply": utt["content"],
                        "gold_score": int(sample["satisfaction_scores"][assistant_turn_idx]),
                        "gold_reason": sample["dissatisfaction_reasons"][assistant_turn_idx],
                        "user": sample.get("user", "unknown"),
                    }
                )
                assistant_turn_idx += 1

            history_window.append(f'{utt["role"]}：{utt["content"]}\n')
            while len(history_window) > 5:
                history_window.pop(0)
    return rows


def build_prompt(row: dict) -> str:
    reason_labels = list(get_reason_to_id().keys())
    reason_labels_text = "、".join(reason_labels)
    dissatisfied_reason_text = "、".join(get_dissatisfied_reasons())
    prompt = (
        "你是一名会进行细粒度对话质量分析的评估员。\n"
        "请基于给定信息先进行推理，再同时预测：\n"
        "1) 当前用户对助手回复的满意度分数（1-5）\n"
        "2) 潜在原因（只有在分数 <=3 时才选择不满意原因；分数 >=4 时必须为“满意”）\n\n"
        f'用户画像：{row["persona"]}\n\n'
        f'任务背景：{row["task_context"]}\n\n'
        f'最近对话历史：{row["history"]}\n\n'
        f'当前助手回复：{row["assistant_reply"]}\n\n'
        f"可选原因标签：{reason_labels_text}\n"
        "原因标签合法性规则：\n"
        f"- 只有当 classification <= 3 时，reason 才能从以下不满意原因中选择：{dissatisfied_reason_text}\n"
        "- 只要 classification >= 4，reason 必须输出“满意”。\n"
        "- 如果 reason 与 classification 不一致，则该输出视为不合法。\n\n"
        "请严格输出一个 JSON 对象，不要输出其他内容：\n"
        '{\n'
        '  "classification": 1-5中的整数,\n'
        '  "reason": "若 classification >= 4 必须输出 满意；若 classification <= 3 只能从其余不满意原因标签中选择一个",\n'
        '  "analysis": "你的详细推理过程"\n'
        "}\n"
    )
    return prompt


def build_messages(prompt: str, model: str) -> list[ChatCompletionMessageParam]:
    if "r1" in model or "reasoner" in model:
        return [
            ChatCompletionUserMessageParam(
                role="user",
                content=prompt,
            )
        ]
    return [
        ChatCompletionSystemMessageParam(
            role="system",
            content="You are a skilled conversational analyst.",
        ),
        ChatCompletionUserMessageParam(
            role="user",
            content=prompt,
        ),
    ]


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


def load_finished_indices(output_jsonl: str) -> set[int]:
    finished: set[int] = set()
    if not os.path.exists(output_jsonl):
        return finished

    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                finished.add(int(obj["sample_id"]))
            except Exception:
                continue
    return finished


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dump_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_rows_from_split(
    split: str = "train",
    split_seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> list[dict]:
    data_list = get_satisfaction_data()
    rows = preprocess_to_rows(data_list)
    users = [row["user"] for row in rows]
    train_idx, valid_idx, test_idx = split_by_user_group_shuffle_split(
        users,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=split_seed,
    )
    idx_map = {
        "train": train_idx,
        "valid": valid_idx,
        "val": valid_idx,
        "test": test_idx,
        "all": list(range(len(rows))),
    }
    if split not in idx_map:
        raise ValueError(f"Invalid split: {split}, expected one of train/valid/val/test/all")
    selected = [rows[i] for i in idx_map[split]]
    logger.info(
        f"Split sizes -> train: {len(train_idx)}, valid: {len(valid_idx)}, test: {len(test_idx)}, selected({split}): {len(selected)}"
    )
    return selected


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


def collect_traces(
    model: str,
    sample_size: int,
    output_jsonl: str,
    seed: int = 42,
    max_workers: int = 8,
    data_split: str = "train",
    split_seed: int = 42,
) -> None:
    rows = get_rows_from_split(split=data_split, split_seed=split_seed)
    rng = random.Random(seed)

    total_available = len(rows)
    if sample_size <= 0 or sample_size > total_available:
        sample_size = total_available
    sampled_rows = rng.sample(rows, k=sample_size)

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    finished_indices = load_finished_indices(output_jsonl)
    output_lock = Lock()
    reason_to_id = get_reason_to_id()
    valid_reasons = set(reason_to_id.keys())
    default_reason = "其它" if "其它" in reason_to_id else next(iter(reason_to_id.keys()))

    logger.info(f"Total candidate rows: {total_available}")
    logger.info(f"Sampled rows: {len(sampled_rows)}")
    logger.info(f"Already finished: {len(finished_indices)}")
    logger.info(f"Output path: {output_jsonl}")

    def process_one(sample_id: int, row: dict) -> tuple[int, dict | None]:
        if sample_id in finished_indices:
            return sample_id, None

        prompt = build_prompt(row)
        messages = build_messages(prompt, model)
        parsed_answer, reasoning_content, raw_content = predict_with_parse(messages, model)
        pred_score = int(parsed_answer.classification)
        pred_reason = parsed_answer.reason.strip()
        normalized_reason = normalize_reason_for_score(
            pred_score,
            pred_reason,
            default_reason=default_reason,
        )
        if not is_reason_valid_for_score(pred_score, pred_reason):
            logger.warning(
                f"Normalized invalid reason/score pair for sample {sample_id}: "
                f"score={pred_score}, raw_reason={pred_reason} -> {normalized_reason}"
            )
        pred_reason = normalized_reason
        if pred_reason not in valid_reasons:
            pred_reason = default_reason

        record = {
            "sample_id": sample_id,
            "model": model,
            "prompt": prompt,
            "messages": messages,
            "prediction": int(pred_score),
            "reason_prediction": pred_reason,
            "reason_prediction_id": int(reason_to_id.get(pred_reason, reason_to_id[default_reason])),
            "analysis": parsed_answer.analysis,
            "gold_score": int(row["gold_score"]),
            "gold_reason": row["gold_reason"],
            "gold_reason_id": int(reason_to_id.get(row["gold_reason"], reason_to_id[default_reason])),
            "meta": {
                "user": row["user"],
                "persona": row["persona"],
                "task_context": row["task_context"],
                "history": row["history"],
                "assistant_reply": row["assistant_reply"],
            },
            "reasoning_content": reasoning_content,
            "raw_content": raw_content,
        }
        return sample_id, record

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_one, i, row): i
            for i, row in enumerate(sampled_rows)
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx = futures[future]
            try:
                _, record = future.result()
                if record is None:
                    continue
                with output_lock:
                    with open(output_jsonl, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"Sample {idx} failed: {e}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-5")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=200,
        help="采样条数；<=0 或超出总量时表示使用全量样本",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="./outputs/api_model_traces.jsonl",
        help="输出文件路径，默认 ./outputs/api_model_traces.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42, help="采样随机种子，默认 42")
    parser.add_argument("--max_workers", type=int, default=8, help="最大工作线程数，默认 8")
    parser.add_argument(
        "--data_split",
        type=str,
        default="train",
        choices=["train", "valid", "val", "test", "all"],
        help="采样来源数据切分；默认 train（与训练脚本一致）",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="数据切分随机种子，默认与训练脚本一致",
    )
    parser.add_argument(
        "--do_reflection",
        action="store_true",
        help="基于已有轨迹文件对错误样本生成反思与重写推理，并输出到新文件",
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default="",
        help="反思模式输入轨迹文件（jsonl）",
    )
    parser.add_argument(
        "--reflection_output_jsonl",
        type=str,
        default="./outputs/api_model_traces_reflection.jsonl",
        help="反思模式输出文件（jsonl）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.do_reflection:
        if not args.input_jsonl:
            raise ValueError("When --do_reflection is enabled, --input_jsonl is required.")
        generate_reflections_from_file(
            model=args.model,
            input_jsonl=args.input_jsonl,
            reflection_output_jsonl=args.reflection_output_jsonl,
            max_workers=args.max_workers,
        )
    else:
        collect_traces(
            model=args.model,
            sample_size=args.sample_size,
            output_jsonl=args.output_jsonl,
            seed=args.seed,
            max_workers=args.max_workers,
            data_split=args.data_split,
            split_seed=args.split_seed,
        )
