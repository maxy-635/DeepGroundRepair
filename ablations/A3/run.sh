#!/bin/bash
export PYTHONPATH=$PYTHONPATH:".../DeepCodeRAG"

module load miniforge3/24.1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate deepcoderag

module load compilers/cuda/12.1 
module load cudnn/8.9.5.29_cuda12.x 

STEP="${1:-}"

MODEL_ID="google/gemma-3-12b-it"
CACHE_DIR=".../huggingface/hub"
DLLS=("TensorFlow" "PyTorch" "PaddlePaddle")
EXPERIMENT_ID="models_0623"
MAX_TOKENS="3000"

run_step3() {
  python "./ablations/A3/step3.py" \
    --model_id "$MODEL_ID" \
    --cache_dir "$CACHE_DIR" \
    --dlls "${DLLS[@]}" \
    --experiment_id "$EXPERIMENT_ID" \
    --max_tokens "$MAX_TOKENS"
}

case "$STEP" in
  step_three)
    run_step3
    ;;
  all)
    run_step3
    ;;
  *)
    echo "用法: bash ablations/A3/run_hf.sh [step_three|all]"
    exit 1
    ;;
esac
