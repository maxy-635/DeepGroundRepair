#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" evaluation/root_cause_location_accuracy/root_cause_evaluator.py \
  --gold_annotations evaluation/root_cause_location_accuracy/root_cause_gold/gold_annotations.csv \
  --predictions evaluation/root_cause_location_accuracy/predictions/root_cause_predictions.csv \
  --report_root evaluation/root_cause_location_accuracy/report
