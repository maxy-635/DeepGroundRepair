#!/bin/bash

export PYTHONPATH=$PYTHONPATH:".../DeepCodeRAG" 
module load miniforge3/24.1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate deepcoderag


MODEL_IDS='["gemma_3_4b_it","gemma_3_12b_it","gemma_3_27b_it","ministral_3_3b_instruct_2512","ministral_3_14b_instruct_2512","mistral_small_3_2_24b_instruct_2506"]'

STEP_ID="step_three"
DLLS='["TensorFlow","PyTorch","PaddlePaddle"]'
EXPERIMENT_ID="****"

python "./evaluation/static_checking/syntatic_checking_main.py" \
     --step_id "$STEP_ID"\
     --model_ids "$MODEL_IDS" \
     --dlls "$DLLS" \
     --experiment_id "$EXPERIMENT_ID"

python "./evaluation/static_checking/messages_statistics.py" \
     --step_id "$STEP_ID"\
     --model_ids "$MODEL_IDS" \
     --dlls "$DLLS" \
     --experiment_id "$EXPERIMENT_ID"
