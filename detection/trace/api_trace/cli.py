from __future__ import annotations

from argparse import ArgumentParser

from .collection import collect_traces
from .reflection import generate_reflections_from_file


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




def main() -> None:
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
