from __future__ import annotations

from pathlib import Path
from typing import Any
from loguru import logger
from ablations import common
from step_two.shape_annotation.comment_cleaner import CommentCleaner
from step_two.shape_annotation.shape_annotator import ShapeAnnotation
from step_two.tensor_shape_trace.tensor_shape_tracer import TensorShapeTrace


NO_RUNTIME_ERROR_TYPE = "No_Runtime_Error"
ROOT_CAUSE_FREE_TYPE = "Shape_Annotated_No_Root_Cause"
SHAPE_ANNOTATION_UNAVAILABLE_TYPE = "Shape_Annotation_Unavailable"


class ShapeAnnotationOnlyPipeline:
    """Implement the shape annotation only pipeline component."""

    def __init__(self, origin_pyfile: str | Path, annotated_pyfile: str | Path, dll_type: str):
        self.origin_pyfile = Path(origin_pyfile)
        self.annotated_pyfile = Path(annotated_pyfile)
        self.dll_type = dll_type.strip().lower()

    def build_result(
        self, final_code: str, runtime_error_info: dict[str, Any] | None,
        root_cause_type: str, shape_annotation: bool
    ) -> dict[str, Any]:
        """Build result."""

        return {
            "runtime_error_info": runtime_error_info,
            "root_cause": {
                "root_cause_type": root_cause_type,
            },
            "shape_annotation": shape_annotation,
            "final_code": final_code,
        }

    def annotate_shapes(
        self, origin_code: str, traced_shapes: dict[int, Any], initial_input_shapes: dict[int, Any] | None = None,
    ) -> str:
        """Annotate shapes."""

        cleaner = CommentCleaner(self.dll_type)
        cleaned_code = cleaner.remove_comments_in_target_method(origin_code)
        return ShapeAnnotation().annotation(
            cleaned_code,
            traced_shapes,
            initial_input_shapes=initial_input_shapes,
            root_cause=None,
        )

    def trace_and_annotate(self, origin_code: str) -> dict[str, Any]:
        """Trace and annotate."""
        try:
            trace_result = TensorShapeTrace(
                python_code=origin_code,
                filename=str(self.origin_pyfile),
            ).main()
            final_code = self.annotate_shapes(
                origin_code=origin_code,
                traced_shapes=trace_result["traced_shapes"],
                initial_input_shapes=trace_result.get("initial_input_shapes", {}),
            )
            runtime_error_info = trace_result["runtime_error_info"]
            root_cause_type = NO_RUNTIME_ERROR_TYPE if runtime_error_info is None else ROOT_CAUSE_FREE_TYPE
            return self.build_result(
                final_code=final_code,
                runtime_error_info=runtime_error_info,
                root_cause_type=root_cause_type,
                shape_annotation=True,
            )

        except Exception as exc:
            logger.error(f"[A2/Step2] Unable to complete shape annotation: {self.origin_pyfile}, error={exc}")
            return self.build_result(
                final_code=origin_code,
                runtime_error_info={
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
                root_cause_type=SHAPE_ANNOTATION_UNAVAILABLE_TYPE,
                shape_annotation=False,
            )

    def main(self) -> dict[str, Any] | None:
        """Run the main workflow."""
        origin_code = self.origin_pyfile.read_text(encoding="utf-8")
        if not origin_code.strip():
            logger.error(f"[A2/Step2] Source code is empty: {self.origin_pyfile}")
            return None

        result = self.trace_and_annotate(origin_code)
        self.annotated_pyfile.parent.mkdir(parents=True, exist_ok=True)
        self.annotated_pyfile.write_text(result["final_code"], encoding="utf-8")
        common.save_json(result, self.annotated_pyfile.with_suffix(".json"))
        return result

    @classmethod
    def run_code_dir(
        cls, source_code_root: str | Path, target_step2_root: str | Path, dll: str,
    ) -> list[Path]:
        """Run code dir."""
        source_root = Path(source_code_root)
        target_root = Path(target_step2_root)
        dll_type = common.normalize_dll(dll)
        saved_json_files: list[Path] = []

        for pyfile in common.iter_pyfiles(source_root):
            annotated_pyfile = common.annotated_path_for_pyfile(pyfile, source_root, target_root)
            pipeline = cls(
                origin_pyfile=pyfile,
                annotated_pyfile=annotated_pyfile,
                dll_type=dll_type,
            )
            logger.info(f"[A2/Step2] Running shape annotation without root-cause guidance: {pyfile}")
            result = pipeline.main()
            if result is None:
                logger.error(f"[A2/Step2] Processing failed: {pyfile}")
                continue
            saved_json_files.append(annotated_pyfile.with_suffix(".json"))

        return saved_json_files
