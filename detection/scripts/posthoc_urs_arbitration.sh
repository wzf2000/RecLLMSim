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
grid_search="${grid_search:-0}"
cv_search="${cv_search:-0}"
candidate_tasks="${candidate_tasks:-advice creative leisure other professional retrieval text}"
max_tasks="${max_tasks:-4}"
search_metric="${search_metric:-balanced}"
n_folds="${n_folds:-5}"
cv_seed="${cv_seed:-42}"

# shellcheck disable=SC2206
strategy_args=($strategies)
# shellcheck disable=SC2206
task_args=($selected_tasks)
# shellcheck disable=SC2206
candidate_task_args=($candidate_tasks)

args=(
    --base_jsonl "$base_jsonl"
    --aux_jsonl "$aux_jsonl"
    --strategies "${strategy_args[@]}"
    --selected_tasks "${task_args[@]}"
    --output_dir "$output_dir"
    --min_samples "$min_samples"
)
[ -n "$output_json" ] && args+=(--output_json "$output_json")
if [ "$grid_search" = "1" ]; then
    args+=(
        --grid_search
        --candidate_tasks "${candidate_task_args[@]}"
        --max_tasks "$max_tasks"
        --search_metric "$search_metric"
    )
fi
if [ "$cv_search" = "1" ]; then
    args+=(
        --cv_search
        --candidate_tasks "${candidate_task_args[@]}"
        --max_tasks "$max_tasks"
        --search_metric "$search_metric"
        --n_folds "$n_folds"
        --cv_seed "$cv_seed"
    )
fi

echo "=========================================="
echo " URS Post-hoc Boundary Arbitration"
echo "  base_jsonl     = $base_jsonl"
echo "  aux_jsonl      = $aux_jsonl"
echo "  strategies     = $strategies"
echo "  selected_tasks = $selected_tasks"
echo "  output_dir     = $output_dir"
echo "  output_json    = ${output_json:-not saved}"
echo "  grid_search    = $grid_search"
echo "  cv_search      = $cv_search"
if [ "$grid_search" = "1" ]; then
    echo "  candidate_tasks= $candidate_tasks"
    echo "  max_tasks      = $max_tasks"
    echo "  search_metric  = $search_metric"
fi
if [ "$cv_search" = "1" ]; then
    echo "  candidate_tasks= $candidate_tasks"
    echo "  max_tasks      = $max_tasks"
    echo "  search_metric  = $search_metric"
    echo "  n_folds        = $n_folds"
    echo "  cv_seed        = $cv_seed"
fi
echo "=========================================="

python eval/urs_posthoc_arbitration.py "${args[@]}"
