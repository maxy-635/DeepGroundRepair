#!/bin/bash

export PYTHONPATH=$PYTHONPATH:".../DeepCodeRAG"      

#load your CUDA and cuDNN modules here
module load compilers/cuda/12.1 
module load cudnn/8.9.5.29_cuda12.x 

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

PYTHON_FILE="./step_one/retriever4generation.py"

python "$PYTHON_FILE" \
     --model_id "google/gemma-3-4b-it" \
     --benchmark "./benchmark/DeepEval/" \
     --step_id "step_one" \
     --dlls '["PyTorch"]' \
     --repeats 1 \
     --experiment_id "test" \