"""Generic LLM-as-judge baselines for personalized satisfaction prediction.

These baselines intentionally do not use per-user profile, user memory, or
source-task history. They score the same personalized target turns as the main
pipeline and emit JSONL records compatible with eval/personalized.py.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock

from loguru import logger
from openai import OpenAI
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_fixed
from tqdm import tqdm

_DETECTION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DETECTION_DIR not in sys.path:
    sys.path.insert(0, _DETECTION_DIR)

from lib.llm import client as default_client
from lib.personalized_data import (
    PersonalizedSample,
    SessionData,
    build_personalized_samples,
    dataset_stats,
)
from lib.satisfaction_constants import normalize_reason_for_score


VARIANTS = [
    "zero_shot",
    "few_shot_global",
    "task_rubric",
    "prometheus_rubric",
]

TASK_RUBRICS: dict[str, str] = {
    "旅行规划": (
        "旅行规划应覆盖行程节奏、地点顺序、交通衔接、住宿区域、预算/时间约束、"
        "避坑提醒和可执行细节；若只是泛泛罗列景点或忽略用户约束，应降低评分。"
    ),
    "礼物准备": (
        "礼物建议应匹配对象关系、预算、场景、偏好和实用性，给出可购买的具体选项、"
        "选择理由与备选方案；若过于模板化或不贴合收礼人，应降低评分。"
    ),
    "菜谱规划": (
        "菜谱规划应包含食材、步骤、火候/时间、替代方案、口味或健康限制，并让用户能直接执行；"
        "若步骤模糊、缺少关键细节或不符合需求，应降低评分。"
    ),
    "技能学习规划": (
        "学习规划应包含阶段目标、资源、练习任务、时间安排、反馈方式和可衡量里程碑；"
        "若只给抽象建议、缺少路径或忽略用户基础，应降低评分。"
    ),
}


@dataclass
class JudgeTurn:
    sample: PersonalizedSample
    session: SessionData
    session_file: str
    turn_idx: int
    dialogue_before: list[dict]
    assistant_reply: str
    gold_score: int
    gold_reason: str

    @property
    def sample_id(self) -> str:
        return (
            f"{self.sample.user}__{self.sample.target_task}__"
            f"{self.session_file}__turn_{self.turn_idx}"
        )


_client = default_client


def configure_client(base_url: str = "", api_key: str = "") -> None:
    global _client
    if base_url:
        _client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")


def _strip_model_wrappers(text: str) -> str:
    content = (text or "").strip()
    if content.startswith("<think>") and "</think>" in content:
        content = content.split("</think>", 1)[1].strip()
    if content.startswith("```json") and content.endswith("```"):
        content = content[7:-3].strip()
    elif content.startswith("```") and content.endswith("```"):
        content = content[3:-3].strip()
    return content


def _extract_json_object(text: str) -> dict:
    content = _strip_model_wrappers(text)
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", content, flags=re.S)
    if not match:
        raise ValueError(f"no JSON object found: {content[:200]}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("JSON payload is not an object")
    return parsed


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    before_sleep=before_sleep_log(logger, log_level=40),
)
def _call_judge(messages: list[dict], model: str, temperature: float, timeout: int) -> dict:
    response = _client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        timeout=timeout,
    ).choices[0].message
    raw = response.content or ""
    return _extract_json_object(raw)


def _assistant_turns(sample: PersonalizedSample) -> list[JudgeTurn]:
    turns: list[JudgeTurn] = []
    for session in sample.target_sessions:
        session_file = os.path.basename(session.file_path)
        dialogue_before: list[dict] = []
        assistant_idx = 0
        for utt in session.history:
            if utt.get("role") != "assistant":
                dialogue_before.append({"role": utt.get("role", ""), "content": utt.get("content", "")})
                continue
            if assistant_idx >= len(session.satisfaction_scores):
                break
            turns.append(
                JudgeTurn(
                    sample=sample,
                    session=session,
                    session_file=session_file,
                    turn_idx=assistant_idx,
                    dialogue_before=list(dialogue_before),
                    assistant_reply=utt.get("content", ""),
                    gold_score=int(session.satisfaction_scores[assistant_idx]),
                    gold_reason=session.dissatisfaction_reasons[assistant_idx],
                )
            )
            dialogue_before.append({"role": "assistant", "content": utt.get("content", "")})
            assistant_idx += 1
    return turns


def _format_dialogue(messages: list[dict], max_chars: int) -> str:
    lines = []
    for msg in messages:
        role = "用户" if msg.get("role") == "user" else "助手"
        lines.append(f"{role}: {msg.get('content', '')}")
    text = "\n".join(lines).strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _format_turn(turn: JudgeTurn, max_context_chars: int) -> str:
    before = _format_dialogue(turn.dialogue_before, max_context_chars)
    return f"""\
任务类型：{turn.sample.target_task}
任务背景：{turn.session.task_context}

对话历史（待评分回复之前）：
{before or "（无）"}

待评分助手回复：
{turn.assistant_reply}
"""


def _score_rubric() -> str:
    return """\
1分：明显没有满足用户需求，严重不相关、不可执行、误导或遗漏关键约束。
2分：只满足少量需求，存在明显缺漏、泛泛而谈或实用性很弱。
3分：部分满足需求，但仍有较明显不足；处在不满意与满意边界的偏不满意侧。
4分：基本满足需求，结构清楚且可执行，仅有轻微不足；处在满意侧。
5分：高度满足或超出需求，具体、贴合、完整、可直接使用。"""


def _output_contract() -> str:
    return """\
只输出 JSON 对象，不要输出其他文字：
{
  "score": 1到5的整数,
  "reason": "若score<=3，从[其它, 不够多样, 不可用, 不够细致, 不满足需求]中选一个；若score>=4，必须为满意",
  "analysis": "一句话说明评分依据"
}"""


def _system_prompt(variant: str) -> str:
    if variant == "prometheus_rubric":
        return (
            "You are a fair response evaluation judge. Evaluate only the response quality "
            "for the given dialogue and rubric. Do not infer hidden user profile or use "
            "personalized memory."
        )
    return (
        "你是一名通用对话满意度评审器。你只能根据当前任务、对话历史和待评分回复判断，"
        "不得使用任何用户画像或跨任务个性化记忆。"
    )


def _prompt_zero_shot(turn: JudgeTurn, max_context_chars: int) -> str:
    return f"""\
请预测用户对下面这轮助手回复的满意度，分数为 1-5。

评分标准：
{_score_rubric()}

{_format_turn(turn, max_context_chars)}

注意：ground truth 中 dissatisfied reason 只在 score<=3 时生效；score>=4 一律视为满意。
{_output_contract()}
"""


def _prompt_few_shot(turn: JudgeTurn, examples: list[JudgeTurn], max_context_chars: int) -> str:
    example_blocks = []
    for i, ex in enumerate(examples, 1):
        example_blocks.append(
            f"""示例{i}
{_format_turn(ex, max_context_chars=1200)}
标注：score={ex.gold_score}, reason={normalize_reason_for_score(ex.gold_score, ex.gold_reason)}
"""
        )
    return f"""\
请参考以下来自训练用户的全局标注示例，预测最后一轮待评分回复的满意度。示例不属于当前用户，不可当作用户记忆。

评分标准：
{_score_rubric()}

全局示例：
{chr(10).join(example_blocks) if example_blocks else "（无）"}

待预测样本：
{_format_turn(turn, max_context_chars)}

注意：ground truth 中 dissatisfied reason 只在 score<=3 时生效；score>=4 一律视为满意。
{_output_contract()}
"""


def _prompt_task_rubric(turn: JudgeTurn, max_context_chars: int) -> str:
    task_rubric = TASK_RUBRICS.get(turn.sample.target_task, "根据任务完成度、约束匹配度、具体性和可执行性评分。")
    return f"""\
请根据当前任务类型的通用 rubric 预测这轮助手回复的用户满意度。

通用 1-5 评分标准：
{_score_rubric()}

任务级 rubric：
{task_rubric}

{_format_turn(turn, max_context_chars)}

注意：ground truth 中 dissatisfied reason 只在 score<=3 时生效；score>=4 一律视为满意。
{_output_contract()}
"""


def _prompt_prometheus(turn: JudgeTurn, max_context_chars: int) -> str:
    return f"""\
### Task Description
Evaluate the assistant response for turn-level user satisfaction in a Chinese task-oriented dialogue.

### Score Rubric
{_score_rubric()}

### Evaluation Criteria
- Consider instruction following, constraint satisfaction, specificity, actionability, correctness, and helpfulness.
- Do not use user profile, personalized memory, or cross-task history.
- A score of 3 is dissatisfied/borderline negative; a score of 4 is satisfied/borderline positive.
- If score >= 4, reason must be "满意". If score <= 3, choose one dissatisfied reason.

### Dialogue And Response
{_format_turn(turn, max_context_chars)}

### Output
{_output_contract()}
"""


def _build_prompt(
    variant: str,
    turn: JudgeTurn,
    few_shot_examples: list[JudgeTurn],
    max_context_chars: int,
) -> str:
    if variant == "zero_shot":
        return _prompt_zero_shot(turn, max_context_chars)
    if variant == "few_shot_global":
        return _prompt_few_shot(turn, few_shot_examples, max_context_chars)
    if variant == "task_rubric":
        return _prompt_task_rubric(turn, max_context_chars)
    if variant == "prometheus_rubric":
        return _prompt_prometheus(turn, max_context_chars)
    raise ValueError(f"Unknown variant: {variant}")


def _round_score(value: object) -> int:
    try:
        return max(1, min(5, int(math.floor(float(value) + 0.5))))
    except Exception:
        return 3


def _make_record(turn: JudgeTurn, pred: dict, model: str, variant: str, parse_ok: bool) -> dict:
    pred_score = _round_score(pred.get("score", 3))
    pred_reason = normalize_reason_for_score(pred_score, str(pred.get("reason", "")))
    return {
        "sample_id": turn.sample_id,
        "user": turn.sample.user,
        "target_task": turn.sample.target_task,
        "target_file": turn.session_file,
        "turn_idx": turn.turn_idx,
        "source_chat_model": turn.session.chat_model,
        "gold_score": turn.gold_score,
        "gold_reason": normalize_reason_for_score(turn.gold_score, turn.gold_reason),
        "pred_score": pred_score,
        "pred_reason": pred_reason,
        "reason_prediction": pred_reason,
        "analysis": str(pred.get("analysis", "")),
        "model": f"generic_llm_judge:{model}",
        "baseline_name": f"generic_llm_judge_{variant}",
        "with_memory": False,
        "memory_version": "none",
        "memory_update_mode": "none",
        "turn_eval_prompt_version": f"generic_llm_judge_{variant}",
        "parse_ok": parse_ok,
    }


def _load_finished(path: str) -> dict[str, dict]:
    records: dict[str, dict] = {}
    if not path or not os.path.exists(path):
        return records
    with open(path, encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            record = json.loads(line)
            sid = record.get("sample_id")
            if sid:
                records[str(sid)] = record
    return records


def _append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def _balanced_examples(samples: list[PersonalizedSample], n: int, seed: int) -> list[JudgeTurn]:
    if n <= 0:
        return []
    by_score: dict[int, list[JudgeTurn]] = {i: [] for i in range(1, 6)}
    for sample in samples:
        for turn in _assistant_turns(sample):
            by_score[turn.gold_score].append(turn)
    rng = random.Random(seed)
    selected: list[JudgeTurn] = []
    per_score = max(1, math.ceil(n / 5))
    for score in range(1, 6):
        pool = by_score[score]
        rng.shuffle(pool)
        selected.extend(pool[:per_score])
    rng.shuffle(selected)
    return selected[:n]


def collect_predictions(
    samples: list[PersonalizedSample],
    examples: list[JudgeTurn],
    model: str,
    variant: str,
    output_jsonl: str,
    max_workers: int,
    max_context_chars: int,
    temperature: float,
    timeout: int,
) -> list[dict]:
    turns = [turn for sample in samples for turn in _assistant_turns(sample)]
    finished = _load_finished(output_jsonl)
    records: dict[str, dict] = dict(finished)
    lock = Lock()

    def process(turn: JudgeTurn) -> dict:
        if turn.sample_id in finished:
            return finished[turn.sample_id]
        prompt = _build_prompt(variant, turn, examples, max_context_chars)
        try:
            pred = _call_judge(
                [
                    {"role": "system", "content": _system_prompt(variant)},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=temperature,
                timeout=timeout,
            )
            record = _make_record(turn, pred, model=model, variant=variant, parse_ok=True)
        except Exception as exc:
            logger.warning(f"Generic judge failed: {turn.sample_id}: {exc}")
            record = _make_record(
                turn,
                {"score": 3, "reason": "其它", "analysis": f"parse/call failed: {exc}"},
                model=model,
                variant=variant,
                parse_ok=False,
            )
        with lock:
            if turn.sample_id not in records:
                records[turn.sample_id] = record
                _append_jsonl(output_jsonl, record)
        return record

    pending = [turn for turn in turns if turn.sample_id not in finished]
    logger.info(f"Total turns={len(turns)}, finished={len(finished)}, pending={len(pending)}")
    if pending:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process, turn) for turn in pending]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Generic judge {variant}"):
                future.result()

    return [records[turn.sample_id] for turn in turns if turn.sample_id in records]


def write_metrics(records: list[dict], path: str, config: dict) -> None:
    if not path:
        return
    from eval.personalized import compute_boundary_metrics, compute_global_metrics

    gold = [float(r["gold_score"]) for r in records]
    pred = [float(r["pred_score"]) for r in records]
    metrics = {
        "global": compute_global_metrics(gold, pred),
        "boundary": compute_boundary_metrics(records),
        "config": config,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2)


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Run generic LLM-as-judge baselines.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--base_url", type=str, default="")
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--variant", choices=VARIANTS, default="zero_shot")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--train_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_history_sessions", type=int, default=1)
    parser.add_argument("--target_tasks", nargs="*", default=None)
    parser.add_argument("--limit_users", type=int, default=0)
    parser.add_argument("--limit_turns", type=int, default=0)
    parser.add_argument("--few_shot_n", type=int, default=10)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--max_context_chars", type=int, default=6000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--output_jsonl", type=str, default="")
    parser.add_argument("--metrics_json", type=str, default="")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    configure_client(args.base_url, args.api_key)

    samples = build_personalized_samples(
        split=args.split,
        train_ratio=args.train_ratio,
        seed=args.seed,
        min_history_sessions=args.min_history_sessions,
        target_tasks=args.target_tasks,
    )
    if args.limit_users > 0:
        allowed_users = sorted({sample.user for sample in samples})[: args.limit_users]
        allowed = set(allowed_users)
        samples = [sample for sample in samples if sample.user in allowed]
    if args.limit_turns > 0:
        kept: list[PersonalizedSample] = []
        total = 0
        for sample in samples:
            if total >= args.limit_turns:
                break
            kept.append(sample)
            total += sample.n_target_turns
        samples = kept

    train_samples = build_personalized_samples(
        split="train" if args.split != "train" else "test",
        train_ratio=args.train_ratio,
        seed=args.seed,
        min_history_sessions=args.min_history_sessions,
        target_tasks=args.target_tasks,
    )
    examples = _balanced_examples(train_samples, args.few_shot_n, args.seed) if args.variant == "few_shot_global" else []

    model_tag = args.model.replace("/", "_").replace(":", "_")
    if not args.output_jsonl:
        args.output_jsonl = (
            f"outputs/personalized/generic_llm_judge_{args.variant}_{model_tag}_{args.split}.jsonl"
        )
    if not args.metrics_json:
        base, _ = os.path.splitext(args.output_jsonl)
        args.metrics_json = base + "_metrics.json"

    logger.info(f"Samples: {dataset_stats(samples)}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Backend: {'custom @ ' + args.base_url if args.base_url else 'default API client'}")
    logger.info(f"Variant: {args.variant}, examples={len(examples)}")

    records = collect_predictions(
        samples=samples,
        examples=examples,
        model=args.model,
        variant=args.variant,
        output_jsonl=args.output_jsonl,
        max_workers=args.max_workers,
        max_context_chars=args.max_context_chars,
        temperature=args.temperature,
        timeout=args.timeout,
    )
    write_metrics(records, args.metrics_json, vars(args))
    logger.info(f"Wrote predictions: {args.output_jsonl}")
    logger.info(f"Wrote metrics: {args.metrics_json}")


if __name__ == "__main__":
    main()
