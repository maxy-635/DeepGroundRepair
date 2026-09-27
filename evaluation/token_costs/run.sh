#!/bin/bash

export PYTHONPATH=$PYTHONPATH:".../DeepCodeRAG" 
module load miniforge3/24.1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate deepcoderag

PYTHON_FILE="./evaluation/token_costs/token_cost_evaluator.py"

LLMs='["gemma_3_4b_it","gemma_3_12b_it","gemma_3_27b_it","ministral_3_3b_instruct_2512","ministral_3_14b_instruct_2512","mistral_small_3_2_24b_instruct_2506"]'

STEP_ID="step_one"
EXPERIMENT_ID="****"

python "$PYTHON_FILE" \
     --LLMs "$LLMs" \
     --step_id "$STEP_ID" \
     --experiment_id "$EXPERIMENT_ID"
