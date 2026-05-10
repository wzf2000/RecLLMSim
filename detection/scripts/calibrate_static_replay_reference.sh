#!/usr/bin/env bash
# Reference-based calibration for static replay scores.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

input="${input:?missing input=outputs/static_replay/...jsonl}"
output="${output:-}"
method="${method:-reference_cdf}"
reference_raw="${reference_raw:-outputs/personalized/Qwen_Qwen3-8B_test_none.jsonl}"
reference_calibrated="${reference_calibrated:-}"
memory_cache_dir="${memory_cache_dir:-outputs/personalized/memory_cache}"
memory_model="${memory_model:-Qwen/Qwen3-8B}"
min_history_turns="${min_history_turns:-5}"

args=(
    --input_jsonl "$input"
    --method "$method"
    --reference_raw_jsonl "$reference_raw"
    --memory_cache_dir "$memory_cache_dir"
    --memory_model "$memory_model"
    --min_history_turns "$min_history_turns"
)

[ -n "$output" ] && args+=(--output_jsonl "$output")
[ -n "$reference_calibrated" ] && args+=(--reference_calibrated_jsonl "$reference_calibrated")

echo "=========================================="
echo " Static Replay Reference Calibration"
echo "  input      = $input"
echo "  method     = $method"
echo "  reference  = $reference_raw"
echo "  output     = ${output:-auto}"
echo "=========================================="

python eval/static_replay_reference_calibrate.py "${args[@]}"
