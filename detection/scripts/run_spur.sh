#!/usr/bin/env bash
# 运行 SPUR 满意度二分类估计 (Lin et al., ACL 2024)
#
# 三阶段：
#   Phase 1 — 从训练集对话中提取 rubric 候选（每标签最多 150 条样本）
#   Phase 2 — 归纳为 10 条代表性 rubric（SAT/DSAT 各 10 条）
#   Phase 3 — 用 rubric 对测试集进行分类
#
# 缓存机制：
#   每个阶段结果自动保存，断点续跑使用 --skip_phase1 / --skip_phase2
#   仅重新计算指标使用 --only_eval
#
# 用法示例：
#   完整运行：   bash scripts/run_spur.sh
#   跳过提取：   bash scripts/run_spur.sh --skip_phase1 --skip_phase2
#   仅看指标：   bash scripts/run_spur.sh --only_eval

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTION_DIR="$(dirname "$SCRIPT_DIR")"
cd "$DETECTION_DIR"

python spur_satisfaction.py \
  --model gpt-4o \
  --k_rubrics 10 \
  --max_extract_per_label 150 \
  --max_workers 8 \
  --output_dir ./outputs/spur \
  "$@"
