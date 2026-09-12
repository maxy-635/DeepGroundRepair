# Step Three

Step Three consumes Step Two diagnoses and generates repaired programs. Repair handlers cover direct repair, unresolved crashes, and tensor-merge mismatches.

## Run

Configure `run_bashes/step_three/run_main_hf.sh`, then run:

```bash
bash run_bashes/step_three/run_main_hf.sh
```

The entry point is `step_three/main.py`.

## Output

```text
response/<model>/step_three/<framework>/<task>/
```

Model loading is implemented in `step_three/llm_runtime.py`; handler selection is implemented under `step_three/handlers/`.
