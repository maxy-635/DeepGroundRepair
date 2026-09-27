# Semantic Consistency

Measures whether repair adds, removes, or preserves effective code lines between Step One and Step Three.

```bash
python evaluation/semantic_consistency/line_count_change_checker.py
```

The script reads the current `response/` layout and writes reports under `evaluation/semantic_consistency/report/`.
