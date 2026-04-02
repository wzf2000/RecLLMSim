#!/usr/bin/env bash
# 运行 SPUR 满意度二分类估计 (Lin et al., ACL 2024)
#
# 四阶段：
#   Phase 1 — 从训练集对话中提取 rubric 候选（每标签最多 150 条样本）
#   Phase 2 — 归纳为 10 条代表性 rubric（SAT/DSAT 各 10 条）
#   Phase 3 — 用 rubric 对测试集进行分类（直接 LLM 判断）
#   Phase 4 — （可选）提取 ada-002 embedding + 训练 LogisticRegression 分类器
#             输出三种变体：rubric_only / embedding_only / combined
#
# 缓存机制：
#   每个阶段结果自动保存，断点续跑使用 --skip_phase1 / --skip_phase2
#   仅重新计算指标使用 --only_eval
#
# 用法示例：
#   完整运行（仅直接LLM判断）：  bash scripts/run_spur.sh
#   启用 embedding 分类器：       bash scripts/run_spur.sh --use_embeddings
#   跳过提取直接评估：            bash scripts/run_spur.sh --skip_phase1 --skip_phase2 --use_embeddings
#   仅看 LLM 判断指标：           bash scripts/run_spur.sh --only_eval

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTION_DIR="$(dirname "$SCRIPT_DIR")"
cd "$DETECTION_DIR"
export PYTHONPATH="$DETECTION_DIR${PYTHONPATH:+:$PYTHONPATH}"

python eval/spur.py \
  --model gpt-4o \
  --k_rubrics 10 \
  --max_extract_per_label 150 \
  --max_workers 8 \
  --output_dir ./outputs/spur \
  "$@"
