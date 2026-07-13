#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-bert-base-chinese}"
SEEDS="${SEEDS:-13,21,42,87,100}"
ITEMS="${ITEMS:-Personality,Daily Interests and Hobbies,Travel Habits,Dining Preferences,Spending Habits}"
DATA_VERSION="${DATA_VERSION:-2}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-512}"
RATIO="${RATIO:-0.1}"
TOPK="${TOPK:-6}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export REQUIRE_CUDA

cd "${PROFILE_DIR}"

"${PYTHON_BIN}" - <<'PY'
import os
import torch
import transformers
import accelerate
from packaging.version import Version

print(f'torch={torch.__version__}')
print(f'transformers={transformers.__version__}')
print(f'accelerate={accelerate.__version__}')
print(f'cuda_available={torch.cuda.is_available()}')
print(f'cuda_device_count={torch.cuda.device_count()}')
if Version(transformers.__version__) < Version('4.42'):
    raise SystemExit('transformers>=4.42 is required.')
if Version(accelerate.__version__) < Version('0.26'):
    raise SystemExit('accelerate>=0.26 is required.')
if os.environ['REQUIRE_CUDA'] == '1' and not torch.cuda.is_available():
    raise SystemExit('CUDA is required. Set REQUIRE_CUDA=0 only for an intentional CPU run.')
if torch.cuda.is_available():
    print(f'cuda_device={torch.cuda.get_device_name(0)}')
PY

RUN_ARGS=(
    --family lm
    --model "${MODEL}"
    --seeds "${SEEDS}"
    --items "${ITEMS}"
    --python "${PYTHON_BIN}"
    --data_version "${DATA_VERSION}"
    --ratio "${RATIO}"
    --topk "${TOPK}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --max_length "${MAX_LENGTH}"
    --include_agent_only
)

if [[ "${FORCE:-0}" == "1" ]]; then
    RUN_ARGS+=(--force)
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    RUN_ARGS+=(--dry_run)
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_repeated.py" "${RUN_ARGS[@]}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "Dry run completed; no experiments or summaries were written."
    exit 0
fi

SUMMARY_DIR="${PROFILE_DIR}/output/repeated"
mkdir -p "${SUMMARY_DIR}"
for METRIC in recall_3 hit_rate_3 f1_micro f1_macro map_macro; do
    "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_repeated.py" \
        --model "${MODEL}" \
        --seeds "${SEEDS}" \
        --items "${ITEMS}" \
        --metric "${METRIC}" \
        --ratio "${RATIO}" \
        --topk "${TOPK}" \
        --include_agent_only \
        --output "${SUMMARY_DIR}/${MODEL//\//__}_hot${TOPK}_ratio${RATIO}_${METRIC}.csv"
done

echo "Completed. Summaries are in ${SUMMARY_DIR}."
