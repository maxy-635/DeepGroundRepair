# Shape-Mismatch Repair Evaluation

Computes shape-mismatch repair success from Step Two and final repair sidecar JSON files. Samples are paired by task ID and normalized filename.

```bash
bash evaluation/shape_mismatch_repair/run.sh
```

The direct entry point is `shape_mismatch_evaluator.py`:

```bash
python evaluation/shape_mismatch_repair/shape_mismatch_evaluator.py \
  --initial_json_root <step-two-sidecars> \
  --final_json_root <final-sidecars> \
  --output_json <report.json>
```

Reports are written under `evaluation/shape_mismatch_repair/report/`. Both input trees must contain matching sidecar JSON files.
