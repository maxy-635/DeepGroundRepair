#!/bin/bash
export PYTHONPATH=$PYTHONPATH:".../DeepCodeRAG"   

module load miniforge3/24.1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate deepcoderag

PYTHON_FILE="./step_two/tensor_shape_debug.py"

python "$PYTHON_FILE" \
     --model_id "google/gemma-3-4b-it" \
     --dlls '["TensorFlow","PyTorch","PaddlePaddle"]' \
     --experiment_id "****"