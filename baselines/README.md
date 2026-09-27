# Baselines

Four baseline configurations are included:

- B1: zero-shot code generation.
- B2: Hybrid-RAG code generation.
- B3: B1 followed by traceback-based self-debugging.
- B4: B2 followed by traceback-based self-debugging.

## Run

Configure each Hugging Face runner, then execute B1 and B2 before their dependent repair baselines:

```bash
bash baselines/B1/run_hf.sh
bash baselines/B2/run_hf.sh
bash baselines/B3/run_hf.sh
bash baselines/B4/run_hf.sh
```

`baselines/run_main.sh` provides the repository's batch wrapper.

## Output

```text
baselines/results/<B>/response/<model>/<framework>/<task>/
```

Shared generation, validation, retrieval, and prompt components are under `baselines/modules/`.
