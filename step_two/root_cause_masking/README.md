# Root-Cause Masking

This package rewrites diagnosed API calls into repair prompts while preserving source layout.

- `root_cause_api_masking.py`: coordinates source edits.
- `root_cause_preprocessor.py`: normalizes supported root-cause structures.
- `api_call_rewriter.py`: rewrites constructor and forward-local calls.
- `api_call_rewriter_utils.py`: shared parsing and formatting helpers.
- `general_crash_linear_masking.py`: handles supported linear-layer cases.

The public integration point is `RootCauseApiMasker.mask(origin_code, root_cause)`, used by `step_two/tensor_shape_debug.py`. Unsupported or ambiguous rewrites are left unchanged.
