# API Gold Sets

The JSON files provide reference APIs for each supported framework.

Each record contains a task identifier and its gold API list:

```json
{"task_id": "easy_task_1", "gold_apis": ["framework.api"]}
```

Task identifiers must match the retrieval records consumed by `evaluation/retrieval_accuracy/retrieval_evaluator.py`.
