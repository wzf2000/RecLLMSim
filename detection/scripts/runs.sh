# 本文件用于记录所有运行过的脚本
# 注释表示还未运行过

# 0. Collection API traces

model=gpt-5 sample_size=1000 output_jsonl=gpt-5_reasoning_traces_v2 ./scripts/collect_api.sh

model=gpt-5 sample_size=1000 output_jsonl=gpt-5_reasoning_traces_reflection input_jsonl=gpt-5_reasoning_traces_v2 ./scripts/collect_api.sh

# 1. SFT

CUDA_VISIBLE_DEVICES=0 batch_size=2 input_jsonl=gpt-5_reasoning_traces_v2.jsonl output_dir=sft_qwen3_from_gpt5_correct ./scripts/sft.sh

CUDA_VISIBLE_DEVICES=0 batch_size=2 input_jsonl=gpt-5_reasoning_traces_v2.jsonl output_dir=sft_qwen3_from_gpt5_correct_reasoning ./scripts/sft.sh

CUDA_VISIBLE_DEVICES=0 batch_size=2 input_jsonl=gpt-5_reasoning_traces_reflection.jsonl output_dir=sft_qwen3_from_gpt5_reflection ./scripts/sft.sh

CUDA_VISIBLE_DEVICES=0 batch_size=2 input_jsonl=gpt-5_reasoning_traces_reflection.jsonl output_dir=sft_qwen3_from_gpt5_reflection_reasoning ./scripts/sft.sh

# running...
CUDA_VISIBLE_DEVICES=0 batch_size=2 input_jsonl=qwen3_self_distill_traces_v3.jsonl output_dir=sft_qwen3_from_self_distill_v3 ./scripts/sft.sh

# 2. Evaluation SFT

CUDA_VISIBLE_DEVICES=0 checkpoint=sft_qwen3_from_gpt5_correct ./scripts/eval_sft.sh

CUDA_VISIBLE_DEVICES=0 checkpoint=sft_qwen3_from_gpt5_correct_reasoning ./scripts/eval_sft.sh

CUDA_VISIBLE_DEVICES=0 checkpoint=sft_qwen3_from_gpt5_reflection ./scripts/eval_sft.sh

CUDA_VISIBLE_DEVICES=0 checkpoint=sft_qwen3_from_gpt5_reflection_reasoning ./scripts/eval_sft.sh

# CUDA_VISIBLE_DEVICES=0 checkpoint=sft_qwen3_from_self_distill_v3 ./scripts/eval_sft.sh

# 3. Collection self-distill traces

# running...
CUDA_VISIBLE_DEVICES=0 distill_version=v1 sft_checkpoint=sft_qwen3_from_gpt5_correct_reasoning output_jsonl=qwen3_self_distill_traces_v2 num_samples_per_prompt=32 min_reasoning_tokens=20 ./scripts/collect_self_distill.sh

CUDA_VISIBLE_DEVICES=0 distill_version=v2 sft_checkpoint=sft_qwen3_from_gpt5_correct_reasoning output_jsonl=qwen3_self_distill_traces_v3 num_samples_per_prompt=4 min_reasoning_tokens=10 ./scripts/collect_self_distill.sh

# 4. GRPO Training

CUDA_VISIBLE_DEVICES=0,1 batch_size=4 sft_checkpoint=sft_qwen3_from_gpt5_correct ./scripts/grpo.sh

CUDA_VISIBLE_DEVICES=0,1 batch_size=4 sft_checkpoint=sft_qwen3_from_gpt5_correct_reasoning ./scripts/grpo.sh

# CUDA_VISIBLE_DEVICES=0,1 batch_size=4 sft_checkpoint=sft_qwen3_from_gpt5_reflection ./scripts/grpo.sh

CUDA_VISIBLE_DEVICES=0,1 batch_size=4 sft_checkpoint=sft_qwen3_from_gpt5_reflection_reasoning ./scripts/grpo.sh

# CUDA_VISIBLE_DEVICES=0,1 batch_size=4 sft_checkpoint=sft_qwen3_from_self_distill_v3 ./scripts/grpo.sh

# 5. Evaluation GRPO

CUDA_VISIBLE_DEVICES=0 sft_checkpoint=sft_qwen3_from_gpt5_correct ./scripts/eval_grpo.sh

CUDA_VISIBLE_DEVICES=0 sft_checkpoint=sft_qwen3_from_gpt5_correct_reasoning ./scripts/eval_grpo.sh

# CUDA_VISIBLE_DEVICES=0 sft_checkpoint=sft_qwen3_from_gpt5_reflection ./scripts/eval_grpo.sh

CUDA_VISIBLE_DEVICES=0 sft_checkpoint=sft_qwen3_from_gpt5_reflection_reasoning ./scripts/eval_grpo.sh

# CUDA_VISIBLE_DEVICES=0 sft_checkpoint=sft_qwen3_from_self_distill_v3 ./scripts/eval_grpo.sh
