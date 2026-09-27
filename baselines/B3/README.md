# B3: Zero-Shot Self-Debugging

B3 executes B1 drafts and applies traceback-based self-debugging when execution fails. B1 results must exist first.

Configure and run:

```bash
bash baselines/B3/run_hf.sh
```

The entry point is `baselines/B3/main_hf.py`. Results are written to `baselines/results/B3/response/<model>/<framework>/<task>/`.
