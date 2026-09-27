# Root-Cause Location Accuracy

Evaluates Step Two root-cause locations against independently annotated labels.

## Workflow

1. Export a stratified annotation sheet with `export_annotation_sheet.py`.
2. Complete and freeze the gold labels under `root_cause_gold/`.
3. Run the evaluator:

```bash
bash evaluation/root_cause_location_accuracy/run.sh
```

`root_cause_evaluator.py` merges gold labels with method predictions and writes reports under `evaluation/root_cause_location_accuracy/report/`.
