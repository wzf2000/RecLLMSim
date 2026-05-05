#!/usr/bin/env bash
# Evaluate scored static replay responses.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

input_jsonl="${input_jsonl:?missing input_jsonl=outputs/static_replay/...scored.jsonl}"
output_json="${output_json:-}"
n_bootstrap="${n_bootstrap:-1000}"
seed="${seed:-42}"

args=(
    --input_jsonl "$input_jsonl"
    --n_bootstrap "$n_bootstrap"
    --seed "$seed"
)
[ -n "$output_json" ] && args+=(--output_json "$output_json")

echo "=========================================="
echo " Static Replay Evaluation"
echo "  input       = $input_jsonl"
echo "  output_json = ${output_json:-none}"
echo "=========================================="

python eval/static_replay.py "${args[@]}"
