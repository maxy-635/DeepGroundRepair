#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-[1,4,8,16,17,25-31]

export PYTHONPATH=$PYTHONPATH:"/home/bingxing2/home/scx8amp/DeepCodeRAG"
module load miniforge3/24.1
source $(conda info --base)/etc/profile.d/conda.sh
conda activate deepcoderag

module load compilers/cuda/12.1 
module load cudnn/8.9.5.29_cuda12.x 

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

PYTHON_FILE="./step_one/vector_store_manager.py"

python3 "$PYTHON_FILE"