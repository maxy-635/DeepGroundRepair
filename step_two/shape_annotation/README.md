# Shape Annotation

This package inserts runtime tensor-shape information and merge-shape expectations into source code.

- `shape_annotator.py`: inserts traced and initial-input shape annotations.
- `comment_cleaner.py`: removes stale generated shape comments.
- `merge_annotation_utils.py`: formats merge-specific oracle annotations.

`ShapeAnnotation.main` is called by `step_two/tensor_shape_debug.py`. Its inputs are source code, traced shapes, initial input shapes, and the normalized root-cause result. The annotated program is written to the configured Step Two output path.
