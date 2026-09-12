# Merge-Shape Debugging

This package detects tensor-shape failures at multi-branch merge operations and identifies the branch and API parameters responsible for the mismatch.

- `analysis/`: mismatch detection and shape comparison.
- `context/`: AST and runtime-trace context recovery.
- `locator/`: branch, API, parameter, and oracle localization.
- `api_rules.json`: framework-specific mismatch rules.

The main integration point is `MergeMismatchRootCauseLocator.locate`, called by `step_two/tensor_shape_debug.py`. Inputs are source code, traceback data, traced shapes, initial input shapes, and the source path.
