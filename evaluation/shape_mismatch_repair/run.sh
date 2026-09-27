#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

if command -v module >/dev/null 2>&1; then
    module load miniforge3/24.1 || true
fi

if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate deepcoderag
fi

MODELS=(
    "gemma_3_12b_it"
    "ministral_3_14b_instruct_2512"
)

DLLS=(
    "tensorflow"
    "pytorch"
    "paddlepaddle"
)

PYTHON_FILE="${PROJECT_ROOT}/evaluation/shape_mismatch_repair/shape_mismatch_evaluator.py"
OUTPUT_ROOT="${PROJECT_ROOT}/evaluation/shape_mismatch_repair/report/RQ3"

for model in "${MODELS[@]}"; do
    for dll in "${DLLS[@]}"; do
        initial_root="${PROJECT_ROOT}/response/${model}/step_two/${dll}"

        python "$PYTHON_FILE" \
            --initial_json_root "$initial_root" \
            --final_json_root "${PROJECT_ROOT}/response/${model}/step_three/${dll}" \
            --output_json "${OUTPUT_ROOT}/full/${model}/${dll}.json"

        python "$PYTHON_FILE" \
            --initial_json_root "$initial_root" \
            --final_json_root "${PROJECT_ROOT}/ablations/results/A3/step_three/${model}/${dll}" \
            --output_json "${OUTPUT_ROOT}/v3/${model}/${dll}.json"
    done
done
