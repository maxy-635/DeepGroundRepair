# B1: Zero-Shot Generation

B1 generates code directly from each task requirement without retrieval or repair.

Configure and run:

```bash
bash baselines/B1/run.sh
```

The entry point is `baselines/B1/main_hf.py`. Results are written to `baselines/results/B1/response/<model>/<framework>/<task>/`.
