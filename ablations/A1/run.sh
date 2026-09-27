#!/bin/bash
export PYTHONPATH=$PYTHONPATH:".../DeepCodeRAG"

module load miniforge3/24.1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate deepcoderag

module load compilers/cuda/12.1 
module load cudnn/8.9.5.29_cuda12.x

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

STEP="${1:-}"

MODEL_ID="google/gemma-3-12b-it"
CACHE_DIR=".../huggingface/hub"
BENCHMARK="./benchmark/DeepEval/"
DLLS=("TensorFlow" "PyTorch" "PaddlePaddle")
EXPERIMENT_ID="models_0623"
REPEATS=1

run_step1() {
  python "./ablations/A1/step1.py" \
    --model_id "$MODEL_ID" \
    --cache_dir "$CACHE_DIR" \
    --benchmark "$BENCHMARK" \
    --dlls "${DLLS[@]}" \
    --experiment_id "$EXPERIMENT_ID" \
    --repeats "$REPEATS"
}

run_step2() {
  python "./ablations/A1/step2.py" \
    --model_id "$MODEL_ID" \
    --dlls "${DLLS[@]}" \
    --experiment_id "$EXPERIMENT_ID"
}

run_step3() {
  python "./ablations/A1/step3.py" \
    --model_id "$MODEL_ID" \
    --cache_dir "$CACHE_DIR" \
    --dlls "${DLLS[@]}" \
    --experiment_id "$EXPERIMENT_ID"
}

case "$STEP" in
  step_one)
    run_step1
    ;;
  step_two)
    run_step2
    ;;
  step_three)
    run_step3
    ;;
  all)
    run_step1
    run_step2
    run_step3
    ;;
  *)
    echo "用法: bash ablations/A1/run_hf.sh [step_one|step_two|step_three|all]"
    exit 1
    ;;
esac
