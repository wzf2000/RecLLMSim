#!/usr/bin/env bash
# Generate statistical/history baseline JSONL files.
#
# Usage from detection/:
#   bash scripts/run_history_baselines.sh
#
# Environment variables:
#   split                 train / test / all (default: test)
#   baselines             space-separated baseline names; default = all
#   output_dir            output directory
#   output_jsonl          optional path, only valid with one baseline
#   limit_users           optional smoke-test user limit
#   min_history_sessions  default 1
#   target_tasks          optional space-separated RecLLMSim task names

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTION_DIR="$(dirname "$SCRIPT_DIR")"
cd "$DETECTION_DIR"
export PYTHONPATH="$DETECTION_DIR${PYTHONPATH:+:$PYTHONPATH}"

split="${split:-test}"
output_dir="${output_dir:-outputs/personalized/history_baselines}"
output_jsonl="${output_jsonl:-}"
limit_users="${limit_users:-0}"
min_history_sessions="${min_history_sessions:-1}"
seed="${seed:-42}"
train_ratio="${train_ratio:-0.2}"
baselines="${baselines:-}"
target_tasks="${target_tasks:-}"

args=(
  --split "$split"
  --output_dir "$output_dir"
  --limit_users "$limit_users"
  --min_history_sessions "$min_history_sessions"
  --seed "$seed"
  --train_ratio "$train_ratio"
)

if [ -n "$output_jsonl" ]; then
  args+=(--output_jsonl "$output_jsonl")
fi

if [ -n "$baselines" ]; then
  # shellcheck disable=SC2206
  baseline_arr=($baselines)
  args+=(--baselines "${baseline_arr[@]}")
fi

if [ -n "$target_tasks" ]; then
  # shellcheck disable=SC2206
  task_arr=($target_tasks)
  args+=(--target_tasks "${task_arr[@]}")
fi

echo "=========================================="
echo " History/statistical baselines"
echo "  split        = $split"
echo "  baselines    = ${baselines:-all}"
echo "  output_dir   = $output_dir"
echo "  output_jsonl = ${output_jsonl:-none}"
echo "  limit_users  = $limit_users"
echo "=========================================="

python eval/history_baselines.py "${args[@]}"

echo "Done."
