#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25-31]
export PYTHONPATH=$PYTHONPATH:"/home/bingxing2/home/scx8amp/DeepCodeRAG"

module load miniforge3/24.1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate deepcoderag

module load compilers/cuda/12.1 
module load cudnn/8.9.5.29_cuda12.x 

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

PYTHON_FILE="./step_three/main.py"

MODEL_ID="google/gemma-3-4b-it"
MODEL_SAVE_NAME=$(echo "$MODEL_ID" | sed 's#.*/##' | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/_/g; s/^_*//; s/_*$//')


STEP2_RESULT_PATH="response/${MODEL_SAVE_NAME}/step_two"
REPAIR_CODE_SAVE_PATH="./response/${MODEL_SAVE_NAME}/step_three"

python "$PYTHON_FILE" \
     --model_id "$MODEL_ID" \
     --cache_dir "/home/bingxing2/home/scx8amp/huggingface/hub" \
     --step2_result_path "$STEP2_RESULT_PATH" \
     --repair_code_save_path "$REPAIR_CODE_SAVE_PATH" \
     --dlls '["TensorFlow","PyTorch","PaddlePaddle"]' \
     --experiment_id "models_0622"
