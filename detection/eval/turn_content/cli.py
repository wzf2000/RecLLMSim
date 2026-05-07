from __future__ import annotations

from argparse import ArgumentParser

from loguru import logger
from openai import OpenAI

from lib.llm import client as default_client

from . import annotation
from .analysis import command_analyze
from .annotation import command_annotate


client: OpenAI = default_client


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Annotate and evaluate substantive assistant turns.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ann = sub.add_parser("annotate", help="LLM-annotate whether turns contain substantive content.")
    p_ann.add_argument("--result_file", type=str, default="", help="Annotate only sample_ids in this JSONL.")
    p_ann.add_argument("--output_jsonl", type=str, required=True)
    p_ann.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    p_ann.add_argument("--vllm_base_url", type=str, default="")
    p_ann.add_argument("--vllm_api_key", type=str, default="EMPTY")
    p_ann.add_argument("--split", type=str, choices=["train", "test", "all"], default="test")
    p_ann.add_argument("--train_ratio", type=float, default=0.2)
    p_ann.add_argument("--split_seed", type=int, default=42)
    p_ann.add_argument("--min_history_sessions", type=int, default=1)
    p_ann.add_argument("--target_tasks", type=str, nargs="*", default=[])
    p_ann.add_argument("--limit", type=int, default=0)
    p_ann.add_argument("--resume", action="store_true", default=True)
    p_ann.add_argument("--no_resume", action="store_false", dest="resume")
    p_ann.add_argument("--save_every", type=int, default=50)
    p_ann.add_argument("--timeout", type=int, default=120)
    p_ann.add_argument("--max_tokens", type=int, default=256)
    p_ann.add_argument(
        "--use_schema_parse",
        action="store_true",
        help="Use SDK schema parsing instead of the faster raw-JSON route.",
    )
    p_ann.set_defaults(func=command_annotate)

    p_eval = sub.add_parser("analyze", help="Analyze result files by content annotation groups.")
    p_eval.add_argument("--annotations_jsonl", type=str, required=True)
    p_eval.add_argument("--result_files", type=str, nargs="+", required=True, metavar="NAME=PATH")
    p_eval.add_argument("--output_json", type=str, default="")
    p_eval.set_defaults(func=command_analyze)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    global client
    if getattr(args, "vllm_base_url", ""):
        client = OpenAI(base_url=args.vllm_base_url, api_key=args.vllm_api_key)
        annotation.set_client(client)
        logger.info(f"vLLM mode: base_url={args.vllm_base_url}")
    args.func(args)
