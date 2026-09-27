#!/bin/bash

export PYTHONPATH=$PYTHONPATH:".../DeepCodeRAG"
module load miniforge3/24.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deepcoderag

PYTHON_FILE="./evaluation/retrieval_accuracy/retrieval_evaluator.py"

MODEL_IDS='["gemma_3_4b_it","gemma_3_12b_it","gemma_3_27b_it","ministral_3_3b_instruct_2512","ministral_3_14b_instruct_2512","mistral_small_3_2_24b_instruct_2506"]'

STEPS='["step_one"]'
DLLS='["TensorFlow","PyTorch","PaddlePaddle"]'
EXPERIMENT_ID="****"
KS='[10]'
ALIAS_JSON="./evaluation/retrieval_accuracy/api_aliases.json"

python "$PYTHON_FILE" \
  --model_ids "$MODEL_IDS" \
  --step_ids "$STEPS" \
  --dlls "$DLLS" \
  --experiment_id "$EXPERIMENT_ID" \
  --api_aliases_json "$ALIAS_JSON" \
  --ks "$KS"
