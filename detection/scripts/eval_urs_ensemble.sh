#!/usr/bin/env bash
# Evaluate offline ensembles of URS predictor outputs.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

runs="${runs:?missing runs='name=path name2=path2 ...'}"
strategies="${strategies:-mean median majority_sat mean_if_confident}"
output_json="${output_json:-}"
output_jsonl="${output_jsonl:-}"
min_samples="${min_samples:-3}"

# shellcheck disable=SC2206
run_args=($runs)
# shellcheck disable=SC2206
strategy_args=($strategies)

args=(
    --runs "${run_args[@]}"
    --strategies "${strategy_args[@]}"
    --min_samples "$min_samples"
)
[ -n "$output_json" ] && args+=(--output_json "$output_json")
[ -n "$output_jsonl" ] && args+=(--output_jsonl "$output_jsonl")

echo "=========================================="
echo " URS Ensemble Evaluation"
echo "  runs       = $runs"
echo "  strategies = $strategies"
echo "  output_json= ${output_json:-not saved}"
echo "=========================================="

python eval/urs_ensemble.py "${args[@]}"
