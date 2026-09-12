# Step Two

Step Two executes generated programs, traces tensor shapes, classifies runtime failures, locates root causes, and writes structured diagnosis sidecars.

## Run

Configure `run_bashes/step_two/run_main_hf.sh`, then run:

```bash
bash run_bashes/step_two/run_main_hf.sh
```

The entry point is `step_two/tensor_shape_debug.py`. It reads Step One programs from `response/<model>/step_one/<framework>/`.

## Output

```text
response/<model>/step_two/<framework>/<task>/
```

Each processed program has an annotated Python file and a JSON diagnosis consumed by Step Three.
