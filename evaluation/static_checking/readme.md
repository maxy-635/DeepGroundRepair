# Static Checking

Runs Pylint on generated programs and aggregates issue categories.

Install the local plugin, then run the evaluator:

```bash
pip install ./evaluation/static_checking/plugin
bash evaluation/static_checking/run.sh
```

Reports are written under `evaluation/static_checking/report/`.
