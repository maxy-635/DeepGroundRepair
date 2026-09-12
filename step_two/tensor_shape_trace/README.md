# Tensor-Shape Trace

This package combines AST analysis with runtime tracing to collect tensor shapes for generated programs.

- `tensor_shape_tracer.py`: coordinates static and runtime analysis.
- `ast_parser.py`: records assignments, calls, and use-definition relationships.
- `runtime_shapes_collector.py`: captures executed-line shapes and initial inputs.
- `sequential_runtime_failure_tracker.py`: refines failures inside sequential containers.

The main integration point is `TensorShapeTrace.main()`, used by `step_two/tensor_shape_debug.py`. It returns traced shapes, initial input shapes, runtime error information, and AST metadata. Framework dependencies must match the generated program being traced.
