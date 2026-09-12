# Step One

Step One recommends APIs, retrieves API documentation, and generates draft programs for each benchmark task.

## Run

Configure the model and environment values in `run_bashes/step_one/run_main_hf.sh`, then run:

```bash
bash run_bashes/step_one/run_main_hf.sh
```

The entry point is `step_one/retriever4generation.py`. The Whoosh and FAISS indexes under `database/` must be available.

## Output

```text
response/<model>/step_one/<framework>/<task>/
evaluation/retrieval_accuracy/retreived_results/<model>/step_one/<framework>/retrieval_api_list.json
```
