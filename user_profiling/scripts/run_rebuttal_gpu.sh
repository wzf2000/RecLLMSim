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
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"
PRELOAD_MODEL="${PRELOAD_MODEL:-1}"

export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export REQUIRE_CUDA
export MODEL
export PRELOAD_MODEL
export DRY_RUN="${DRY_RUN:-0}"

cd "${PROFILE_DIR}"

"${PYTHON_BIN}" - <<'PY'
import os
import gc
import torch
import transformers
import accelerate
from packaging.version import Version
from transformers import AutoModel, AutoTokenizer

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
for index in range(torch.cuda.device_count()):
    print(f'cuda_device_{index}={torch.cuda.get_device_name(index)}')
if os.environ['PRELOAD_MODEL'] == '1' and os.environ['DRY_RUN'] != '1':
    model_name = os.environ['MODEL']
    print(f'Preloading {model_name} into the shared Hugging Face cache...')
    AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    del model
    gc.collect()
PY

RUN_ARGS=(
    --family lm
    --model "${MODEL}"
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

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
IFS=',' read -r -a SEED_ARRAY <<< "${SEEDS}"
if [[ "${#GPU_ARRAY[@]}" -eq 0 ]]; then
    echo "GPU_IDS must contain at least one GPU index." >&2
    exit 1
fi

declare -a WORKER_SEEDS
for INDEX in "${!SEED_ARRAY[@]}"; do
    SEED="${SEED_ARRAY[$INDEX]// /}"
    WORKER_INDEX=$((INDEX % ${#GPU_ARRAY[@]}))
    if [[ -z "${WORKER_SEEDS[$WORKER_INDEX]:-}" ]]; then
        WORKER_SEEDS[$WORKER_INDEX]="${SEED}"
    else
        WORKER_SEEDS[$WORKER_INDEX]="${WORKER_SEEDS[$WORKER_INDEX]},${SEED}"
    fi
done

WORKER_LOG_DIR="${PROFILE_DIR}/output/repeated/workers"
mkdir -p "${WORKER_LOG_DIR}"
declare -a WORKER_PIDS
declare -a WORKER_NAMES
for INDEX in "${!GPU_ARRAY[@]}"; do
    ASSIGNED_SEEDS="${WORKER_SEEDS[$INDEX]:-}"
    if [[ -z "${ASSIGNED_SEEDS}" ]]; then
        continue
    fi
    GPU_ID="${GPU_ARRAY[$INDEX]// /}"
    WORKER_LOG="${WORKER_LOG_DIR}/gpu_${GPU_ID}.log"
    echo "GPU ${GPU_ID}: seeds ${ASSIGNED_SEEDS}; log ${WORKER_LOG}"
    (
        export CUDA_VISIBLE_DEVICES="${GPU_ID}"
        "${PYTHON_BIN}" "${SCRIPT_DIR}/run_repeated.py" "${RUN_ARGS[@]}" --seeds "${ASSIGNED_SEEDS}"
    ) >"${WORKER_LOG}" 2>&1 &
    WORKER_PIDS+=("$!")
    WORKER_NAMES+=("GPU ${GPU_ID} (seeds ${ASSIGNED_SEEDS})")
done

FAILED=0
for INDEX in "${!WORKER_PIDS[@]}"; do
    if wait "${WORKER_PIDS[$INDEX]}"; then
        echo "Completed: ${WORKER_NAMES[$INDEX]}"
    else
        echo "Failed: ${WORKER_NAMES[$INDEX]}" >&2
        FAILED=1
    fi
done
if [[ "${FAILED}" -ne 0 ]]; then
    echo "At least one GPU worker failed. Inspect ${WORKER_LOG_DIR}/*.log and rerun the same command after fixing the issue." >&2
    exit 1
fi

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
