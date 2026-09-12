# Crash Locator

This package maps a runtime traceback back to the relevant model definition.

- `traceback_parser.py`: extracts user-code frames and crash locations.
- `called_module_extractor.py`: resolves calls such as `self.layer(...)`.
- `root_cause_locator.py`: maps forward calls to initialization definitions.
- `container_refiner.py`: refines failures inside sequential containers.
- `forward_local_locator.py`: handles calls defined directly in the forward method.
- `root_cause_models.py`: defines normalized result structures.

The package is called by `step_two/tensor_shape_debug.py`; it has no standalone experiment runner.
