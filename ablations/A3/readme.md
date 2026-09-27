# A3

A3 reuses main-experiment Step Two diagnoses but removes tensor-shape context from the repair prompt.

## Run

Configure `ablations/A3/run.sh`, then run:

```bash
bash ablations/A3/run.sh step3
```

The Step Two source can be set with `--step2_result_root`. Results are written to `ablations/results/A3/step_three/<model>/<framework>/<task>/`.
