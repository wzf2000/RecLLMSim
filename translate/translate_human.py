import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from loguru import logger
from openai import OpenAI
from tenacity import (  # type: ignore[import]
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

try:
    from tqdm import tqdm  # type: ignore[import]
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


def load_api_config(repo_root: Path) -> dict[str, Any]:
    """
    Prefer repo root `api_config.json` (this repo uses base_url/api_key),
    fall back to RecUserSim config conventions if present.
    """
    candidates = [
        repo_root / "api_config.json",
        repo_root / "RecUserSim" / "config" / "api-config.json",
        repo_root / "RecUserSim" / "config" / "api_config.json",
    ]
    for p in candidates:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            return cfg
    raise FileNotFoundError(
        "未找到 API 配置文件。尝试过：\n" + "\n".join(str(p) for p in candidates)
    )


def init_openai_env(cfg: dict[str, Any]) -> None:
    # This repo's `api_config.json` uses: base_url, api_key
    base_url = cfg.get("base_url") or cfg.get("api_base") or cfg.get("api_base_url")
    api_key = cfg.get("api_key") or cfg.get("key")
    if not api_key:
        raise ValueError("API 配置中缺少 api_key")
    if base_url:
        os.environ["OPENAI_API_BASE"] = str(base_url)
        logger.info(f"设置 OPENAI_API_BASE: {base_url}")
    os.environ["OPENAI_API_KEY"] = str(api_key)


def iter_json_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*.json") if p.is_file()])


def _invoke_chat_completions(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    timeout: int,
) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("环境变量 OPENAI_API_KEY 未设置")

    try:
        base_url = os.environ.get("OPENAI_API_BASE")
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
    except Exception as e:
        raise RuntimeError(f"OpenAI 请求失败: {e}") from e

    if not resp.choices:
        raise RuntimeError("OpenAI 响应缺少 choices")
    content = getattr(resp.choices[0].message, "content", None)
    if content is None:
        raise RuntimeError("OpenAI 响应缺少 message.content")
    return str(content)


SYSTEM_PROMPT = (
    "You are a professional translator.\n"
    "Task: translate the user's text into natural English.\n"
    "Rules:\n"
    "- Preserve the original meaning and tone.\n"
    "- Keep lists, line breaks, punctuation, and any markdown formatting.\n"
    "- Do NOT add explanations.\n"
    "- Output ONLY the English translation.\n"
)


@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_exponential_jitter(exp_base=2, max=30),
    retry=retry_if_exception_type(Exception),
)
def _translate_with_retry(model: str, text: str) -> str:
    out = _invoke_chat_completions(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
        timeout=120,
    )
    if not isinstance(out, str):
        out = str(out)
    return out.strip()


def translate_text(model: str, text: str, max_retries: int = 6) -> str:
    """保持签名兼容，内部使用 tenacity 重试。"""
    if text is None:
        return ""
    text = str(text)
    if text.strip() == "":
        return ""
    try:
        return _translate_with_retry(model, text)
    except Exception as e:
        raise RuntimeError("翻译多次失败，已放弃。") from e


def process_file(
    path: Path,
    llm: str,
    overwrite: bool = False,
    dry_run: bool = False,
    max_turns: int = 0,
) -> tuple[int, int]:
    """
    Returns (turns_total, turns_translated)
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    history = data.get("history")
    if not isinstance(history, list):
        return (0, 0)

    total = 0
    translated = 0
    for turn in history:
        if max_turns and total >= max_turns:
            break
        if not isinstance(turn, dict):
            continue
        if "content" not in turn:
            continue
        total += 1
        if (not overwrite) and isinstance(turn.get("content_en"), str) and turn["content_en"].strip() != "":
            continue
        src = turn.get("content", "")
        logger.debug(f"翻译第 {total} 轮：{src}")
        turn["content_en"] = translate_text(llm, src)
        translated += 1

    if translated > 0 and not dry_run:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            f.write("\n")

    return (total, translated)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="为 human_exp_V2 与 real_human_user 的 history[*].content 添加英文翻译到 content_en"
    )
    parser.add_argument(
        "--repo_root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="仓库根目录（默认：translate/ 的上一级）",
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="翻译模型，如 gpt-4o-mini / gpt-5-mini")
    parser.add_argument("--overwrite", action="store_true", help="是否覆盖已存在的 content_en")
    parser.add_argument("--dry_run", action="store_true", help="只统计不写回文件")
    parser.add_argument("--max_files", type=int, default=0, help="最多处理多少个文件，0 表示不限制")
    parser.add_argument("--max_turns", type=int, default=0, help="每个文件最多翻译多少轮（0 表示不限制）")
    parser.add_argument(
        "--file",
        type=str,
        default="",
        help="只处理某一个文件（相对 repo_root 或绝对路径）",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="每次翻译后额外 sleep 秒数（限流用）")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并行处理文件的线程数（默认：4）",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    cfg = load_api_config(repo_root)
    init_openai_env(cfg)

    llm = args.model

    files: list[Path] = []
    if args.file:
        p = Path(args.file)
        if not p.is_absolute():
            p = repo_root / p
        files = [p.resolve()]
    else:
        targets = [
            repo_root / "human_exp_V2",
            repo_root / "real_human_user",
        ]
        for t in targets:
            files.extend(iter_json_files(t))

    if args.max_files and args.max_files > 0:
        files = files[: args.max_files]

    total_files = len(files)
    logger.info(
        f"将处理 {total_files} 个文件。model={args.model} "
        f"overwrite={args.overwrite} dry_run={args.dry_run} workers={args.workers}"
    )

    if total_files == 0:
        logger.info("没有需要处理的文件。")
        return
    total_turns = 0
    total_translated = 0
    done_files = 0

    workers = max(1, args.workers)

    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=total_files, desc="翻译文件", unit="file")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_file,
                fp,
                llm,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                max_turns=args.max_turns,
            ): fp
            for fp in files
        }

        for fut in as_completed(futures):
            fp = futures[fut]
            try:
                t_total, t_trans = fut.result()
            except Exception as e:
                logger.error(f"处理文件失败：{fp}，错误：{e}")
                t_total, t_trans = 0, 0

            done_files += 1
            total_turns += t_total
            total_translated += t_trans

            if pbar is not None:
                pbar.update(1)
                try:
                    pbar.set_postfix({"translated": total_translated})
                except Exception:
                    # 某些 tqdm 实现不支持 dict postfix，忽略即可
                    pass

            if args.sleep and args.sleep > 0 and t_trans > 0:
                time.sleep(args.sleep)

    if pbar is not None:
        pbar.close()
    logger.success(
        f"完成。共扫描 {total_files} 文件，轮次 {total_turns}，新增翻译 {total_translated}。"
    )


if __name__ == "__main__":
    main()
