# Merge-Mismatch Repair

This package models erroneous branch shapes, retrieves API shape knowledge, solves candidate parameters, and rewrites code for tensor-merge mismatches.

- `buggy_shape_modeling.py`: reconstructs the failing shape context.
- `retrieve4shapeknowledge.py`: retrieves shape formulas.
- `params_solver.py`: solves candidate parameters.
- `shape_simulator.py`: simulates supported shape transformations.
- `helper/code_rewriter.py`: applies AST-based edits.

The package is called by `step_three/handlers/merge_mismatch_handler.py`. The formula table is `dl_api_mapping_shape_formulas.xlsx`.
