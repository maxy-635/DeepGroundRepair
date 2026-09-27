# A2

A2 retains runtime shape tracing and annotations but removes targeted root-cause guidance from repair.

## Run

Configure `ablations/A2/run.sh`, then run:

```bash
bash ablations/A2/run.sh all
bash ablations/A2/run.sh step2
bash ablations/A2/run.sh step3
```

Step Two reads the main Step One output configured by `--draft_root`. Results are written to `ablations/results/A2/<step>/<model>/<framework>/<task>/`.
