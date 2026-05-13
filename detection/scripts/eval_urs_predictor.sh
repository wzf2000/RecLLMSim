#!/usr/bin/env bash
# Evaluate URS satisfaction predictor outputs.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

result_file="${result_file:-}"
baseline_file="${baseline_file:-}"
result_files="${result_files:-}"
output_json="${output_json:-}"
min_samples="${min_samples:-3}"

args=(--min_samples "$min_samples")
[ -n "$result_file" ] && args+=(--result_file "$result_file")
[ -n "$baseline_file" ] && args+=(--baseline_file "$baseline_file")
if [ -n "$result_files" ]; then
    # shellcheck disable=SC2206
    args+=(--result_files $result_files)
fi
[ -n "$output_json" ] && args+=(--output_json "$output_json")

echo "=========================================="
echo " URS Satisfaction Predictor Evaluation"
echo "  result_file  = ${result_file:-not specified}"
echo "  baseline     = ${baseline_file:-not specified}"
echo "  result_files = ${result_files:-not specified}"
echo "  output_json  = ${output_json:-not saved}"
echo "=========================================="

python eval/personalized.py "${args[@]}"
