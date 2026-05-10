#!/usr/bin/env bash
# Pairwise win/tie/lose comparison for static replay scored outputs.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

reference_file="${reference_file:?missing reference_file=outputs/static_replay/...jsonl}"
reference_name="${reference_name:-}"
candidate_files="${candidate_files:?missing candidate_files='name=path ...'}"
tie_margin="${tie_margin:-0}"
output_json="${output_json:-}"

args=(
    --reference_file "$reference_file"
    --tie_margin "$tie_margin"
)
[ -n "$reference_name" ] && args+=(--reference_name "$reference_name")
[ -n "$output_json" ] && args+=(--output_json "$output_json")

# shellcheck disable=SC2206
candidate_arr=($candidate_files)
args+=(--candidate_files "${candidate_arr[@]}")

echo "=========================================="
echo " Static Replay Pairwise Comparison"
echo "  reference = $reference_file"
echo "  tie_margin= $tie_margin"
echo "  output    = ${output_json:-none}"
echo "=========================================="

python eval/static_replay_pairwise.py "${args[@]}"
