# B4: Hybrid-RAG Self-Debugging

B4 executes B2 drafts and applies traceback-based self-debugging when execution fails. B2 results must exist first.

Configure and run:

```bash
bash baselines/B4/run_hf.sh
```

The entry point is `baselines/B4/main_hf.py`. Results are written to `baselines/results/B4/response/<model>/<framework>/<task>/`.
