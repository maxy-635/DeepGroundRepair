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

run_step2() {
  local llm_name
  llm_name=$(python -c "from utils.utils import get_llm_name; print(get_llm_name('$MODEL_ID'))")
  local args=(
    "./ablations/A2/step2.py"
    --model_id "$MODEL_ID"
    --dlls "${DLLS[@]}"
    --experiment_id "$EXPERIMENT_ID"
    --draft_root "./response/${llm_name}/step_one"
  )
  python "${args[@]}"
}

run_step3() {
  python "./ablations/A2/step3.py" \
    --model_id "$MODEL_ID" \
    --cache_dir "$CACHE_DIR" \
    --dlls "${DLLS[@]}" \
    --experiment_id "$EXPERIMENT_ID"
}

case "$STEP" in
  step_two)
    run_step2
    ;;
  step_three)
    run_step3
    ;;
  all)
    run_step2
    run_step3
    ;;
  *)
    echo "用法: bash ablations/A2/run.sh [step_two|step_three|all]"
    exit 1
    ;;
esac
