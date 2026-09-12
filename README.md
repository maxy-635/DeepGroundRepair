# DeepGroundRepair

DeepGroundRepair is the implementation and experiment repository for retrieval-assisted deep-learning code generation, tensor-shape diagnosis, root-cause localization, and guided repair.

## Pipeline

1. `step_one/`: retrieve API documentation and generate draft code.
2. `step_two/`: execute drafts, trace tensor shapes, and locate root causes.
3. `step_three/`: regenerate or repair code from the Step Two diagnosis.

The experiment model families are Gemma and Mistral. The evaluated frameworks are TensorFlow, PyTorch, and PaddlePaddle.

## Run

Install the environment:

```bash
conda env create -f environment.yaml
conda activate deepcoderag
```

Configure the model paths and cluster settings in the relevant script, then run:

```bash
bash run_bashes/step_one/run_main_hf.sh
bash run_bashes/step_two/run_main_hf.sh
bash run_bashes/step_three/run_main_hf.sh
```

Baseline and ablation runners are documented in `baselines/README.md` and `ablations/README.md`. Evaluation commands are documented in `evaluation/README.md`.

## Results

Main results are stored under `response/`; baseline and ablation results are under `baselines/results/` and `ablations/results/`.
