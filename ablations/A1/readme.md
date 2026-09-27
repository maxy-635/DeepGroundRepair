# A1

A1 replaces Step One retrieval-assisted generation with zero-shot generation. Step Two diagnosis and Step Three repair remain unchanged.

## Run

Configure `ablations/A1/run.sh`, then run all stages or one stage:

```bash
bash ablations/A1/run.sh all
bash ablations/A1/run.sh step1
bash ablations/A1/run.sh step2
bash ablations/A1/run.sh step3
```

Results are written to `ablations/results/A1/<step>/<model>/<framework>/<task>/`.
