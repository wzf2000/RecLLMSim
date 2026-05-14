#!/usr/bin/env bash
# Post-hoc boundary arbitration for URS predictor outputs.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

base_jsonl="${base_jsonl:?missing base_jsonl=path/to/base.jsonl}"
aux_jsonl="${aux_jsonl:?missing aux_jsonl=path/to/aux.jsonl}"
strategies="${strategies:-selected_downgrade selected_boundary_aux all_boundary_aux}"
selected_tasks="${selected_tasks:-leisure professional text other}"
output_dir="${output_dir:-outputs/urs/posthoc}"
output_json="${output_json:-}"
min_samples="${min_samples:-3}"

# shellcheck disable=SC2206
strategy_args=($strategies)
# shellcheck disable=SC2206
task_args=($selected_tasks)

args=(
    --base_jsonl "$base_jsonl"
    --aux_jsonl "$aux_jsonl"
    --strategies "${strategy_args[@]}"
    --selected_tasks "${task_args[@]}"
    --output_dir "$output_dir"
    --min_samples "$min_samples"
)
[ -n "$output_json" ] && args+=(--output_json "$output_json")

echo "=========================================="
echo " URS Post-hoc Boundary Arbitration"
echo "  base_jsonl     = $base_jsonl"
echo "  aux_jsonl      = $aux_jsonl"
echo "  strategies     = $strategies"
echo "  selected_tasks = $selected_tasks"
echo "  output_dir     = $output_dir"
echo "  output_json    = ${output_json:-not saved}"
echo "=========================================="

python eval/urs_posthoc_arbitration.py "${args[@]}"
