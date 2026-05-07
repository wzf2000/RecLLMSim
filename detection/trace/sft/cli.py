from __future__ import annotations

from argparse import ArgumentParser

from .training import train_sft


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True, help="轨迹 jsonl 路径")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--output_dir", type=str, default="./ckpts/sft_from_api_traces")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--include_reasoning_content",
        action="store_true",
        help="是否将 reasoning_content 与 raw_content 拼接后作为监督目标",
    )
    parser.add_argument(
        "--think_wrap",
        type=str,
        default="qwen3",
        choices=["qwen3", "none"],
        help=(
            "qwen3：在 assistant 监督目标外按 Qwen3 chat_template 包一层 <think>.../think 标记，"
            "再拼接可见 JSON（与 model.generate 输出对齐）。"
            "none：保持旧版纯文本（reasoning\\n\\nJSON），用于非 Qwen3 基座。"
        ),
    )
    parser.add_argument(
        "--max_history_turns",
        type=int,
        default=5,
        help="history 最多保留的轮次数（与采集端滑动窗口一致）；过长时会从更少轮次尝试以适配 max_length",
    )
    parser.add_argument(
        "--trace_source",
        type=str,
        default="correct_only",
        choices=["correct_only", "correct_plus_reflected_wrong"],
        help=(
            "correct_only：仅首轮预测与 gold 完全一致的轨迹（默认）。"
            "correct_plus_reflected_wrong：在此基础上再并入首轮错误、"
            "且含 collect_api_model_traces.generate_reflections_from_file 所写字段 reflection 的样本；"
            "监督目标为 gold 分数/原因 + reflection.revised_reasoning（与首轮 raw_content 格式一致的 JSON）。"
        ),
    )
    return parser.parse_args()


def run_cli() -> None:
    args = parse_args()
    train_sft(**vars(args))
