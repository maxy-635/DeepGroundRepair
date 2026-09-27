# B2: Hybrid-RAG Generation

B2 retrieves API documentation with sparse and dense retrieval, then generates code from the task and retrieved context.

Configure and run:

```bash
bash baselines/B2/run_hf.sh
```

The entry point is `baselines/B2/main_hf.py`. Results are written to `baselines/results/B2/response/<model>/<framework>/<task>/`.
