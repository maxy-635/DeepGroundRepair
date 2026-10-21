#!/bin/bash
export PYTHONPATH=$PYTHONPATH:".../DeepCodeRAG"   

PYTHON_FILE="./step_two/tensor_shape_debug.py"

python "$PYTHON_FILE" \
     --model_id "google/gemma-3-4b-it" \
     --dlls '["TensorFlow","PyTorch","PaddlePaddle"]'