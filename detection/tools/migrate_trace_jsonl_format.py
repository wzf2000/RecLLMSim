import json
import os
from argparse import ArgumentParser

from loguru import logger


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
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


def _looks_like_json_content(text: str) -> bool:
    s = text.strip()
    if not s:
        return False
    if s.startswith("{") or s.startswith("```json"):
        return True
    return '"classification"' in s and '"reason"' in s


def split_raw_output(raw_output: str) -> tuple[str, str]:
    """
    旧版 raw_output 可能是:
    1) reasoning_content + "\\n\\n" + content
    2) 仅 content
    3) 仅 reasoning（极少见）
    这里做启发式拆分，尽可能恢复为:
    reasoning_content, raw_content
    """
    text = raw_output or ""
    if not text.strip():
        return "", ""

    # 优先尝试从后往前按双换行切，找到最像 content 的末段
    parts = text.split("\n\n")
    if len(parts) >= 2:
        for i in range(len(parts) - 1, 0, -1):
            candidate_content = "\n\n".join(parts[i:]).strip()
            candidate_reasoning = "\n\n".join(parts[:i]).strip()
            if _looks_like_json_content(candidate_content):
                return candidate_reasoning, candidate_content

    # 如果整体看起来像 content，则认为没有 reasoning
    if _looks_like_json_content(text):
        return "", text.strip()

    # 否则保守处理：都当作 reasoning，content 置空
    return text.strip(), ""


def migrate_record(record: dict) -> dict:
    new_record = dict(record)

    # 主结果字段迁移
    if "reasoning_content" not in new_record or "raw_content" not in new_record:
        old_raw_output = new_record.get("raw_output")
        if isinstance(old_raw_output, str):
            reasoning_content, raw_content = split_raw_output(old_raw_output)
            new_record["reasoning_content"] = reasoning_content
            new_record["raw_content"] = raw_content
        else:
            new_record.setdefault("reasoning_content", "")
            new_record.setdefault("raw_content", "")

    # 保留旧字段也可以；如果你希望删掉可在这里 pop
    # new_record.pop("raw_output", None)

    # reflection 子字段迁移
    reflection = new_record.get("reflection")
    if isinstance(reflection, dict):
        if "reasoning_content" not in reflection or "raw_content" not in reflection:
            reflection_raw_output = reflection.get("raw_output")
            if isinstance(reflection_raw_output, str):
                r_reasoning, r_content = split_raw_output(reflection_raw_output)
                reflection["reasoning_content"] = r_reasoning
                reflection["raw_content"] = r_content
            else:
                reflection.setdefault("reasoning_content", "")
                reflection.setdefault("raw_content", "")
        new_record["reflection"] = reflection

    return new_record


def migrate_file(input_jsonl: str, output_jsonl: str) -> None:
    rows = load_jsonl(input_jsonl)
    migrated = [migrate_record(row) for row in rows]
    dump_jsonl(output_jsonl, migrated)
    logger.info(f"Migrated {len(migrated)} records")
    logger.info(f"Input: {input_jsonl}")
    logger.info(f"Output: {output_jsonl}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True, help="旧版轨迹 jsonl 路径")
    parser.add_argument("--output_jsonl", type=str, required=True, help="新版轨迹 jsonl 输出路径")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    migrate_file(args.input_jsonl, args.output_jsonl)
