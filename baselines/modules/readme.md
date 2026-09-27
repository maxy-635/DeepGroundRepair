# Baseline Modules

Shared baseline components:

- `experiment_utils.py`: model loading and result paths.
- `hybrid_rag.py` and `rag/`: B2/B4 retrieval.
- `debugging.py`: generated-code execution and traceback cleanup.
- `no_weight_validation.py`: subprocess validation without downloading pretrained weights.
- `zeroshot_prompt.py`, `hybridrag_prompt.py`, and `self_debugging_prompt.py`: prompt construction.

These modules are imported by B1-B4 and are not independent experiment entry points.
