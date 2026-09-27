#!/bin/bash
export PYTHONPATH=$PYTHONPATH:".../DeepCodeRAG"

module load miniforge3/24.1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate deepcoderag

module load compilers/cuda/12.1 
module load cudnn/8.9.5.29_cuda12.x 


PYTHON_FILE="./baselines/B3/main.py"

python "$PYTHON_FILE" \
     --model_id "google/gemma-3-4b-it" \
     --model_cache_path ".../huggingface/hub" \
     --benchmark "./benchmark/DeepEval/" \
     --dlls '["TensorFlow","PyTorch","PaddlePaddle"]' \
     --experiment_id "****" \
     --repeats 1
