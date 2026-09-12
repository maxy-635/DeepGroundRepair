# Merge-Shape Locator

Locator modules identify the mismatching branch and the API parameters that can restore compatible tensor shapes.

- `target_branch_identifier.py`: selects the mismatching branch.
- `target_api_locator.py`: traces the branch to candidate APIs.
- `target_api_parameter_masker.py`: selects repairable keyword parameters.
- `spatial_dim_processing_helper.py`: applies spatial-dimension rules.
- `merge_shape_oracle_builder.py`: builds the expected-shape oracle.
- `target_api_locator_utils.py`: shared AST and trace helpers.

These modules are used through `MergeMismatchRootCauseLocator`; they are not command-line entry points.
