import os
import json
import argparse
import traceback
from pathlib import Path
from typing import Any, Dict
from loguru import logger
# tensor shape tracing module
from step_two.tensor_shape_trace.tensor_shape_tracer import TensorShapeTrace
# root cause locator module
from step_two.crash_locator.root_cause_locator import TracebackToInitMapper
from step_two.crash_locator.forward_local_locator import ForwardLocalCallLocator
from step_two.crash_locator.root_cause_models import (
        build_general_root_cause_result,
        build_unresolved_root_cause_result,
        get_root_cause_original_call
    )
from step_two.crash_locator.traceback_parser import TracebackParser
# merge mismatch debug module
from step_two.merge_shape_debug.analysis.merge_mismatch_detector import MergeMismatchDetector
from step_two.merge_shape_debug.locator.merge_root_cause_locator import MergeMismatchRootCauseLocator
from step_two.root_cause_masking.root_cause_api_masking import RootCauseApiMasker
# shape annotation module
from step_two.shape_annotation.comment_cleaner import CommentCleaner
from step_two.shape_annotation.shape_annotator import ShapeAnnotation
from step_two.utils import compact_step2_result_for_save, get_all_files, get_llm_name, remove_local_path_info



class TensorShapeDebugPipeline:
    """Implement the tensor shape debug pipeline component."""

    SUPPORTED_DLL_TYPES = {"tensorflow", "pytorch", "paddlepaddle"}

    def __init__(self, origin_pyfile: str, annotated_pyfile: str, dll_type: str):
        """Initialize the instance."""
        self.origin_pyfile = origin_pyfile
        self.annotated_pyfile = annotated_pyfile
        self.dll_type  = dll_type.strip().lower()
        self.root_cause_api_masker = RootCauseApiMasker()

    def locate_root_cause(self, origin_code: str, runtime_error_info: [Dict[str, Any]], 
                        traced_shapes=None, initial_input_shapes=None,
                    ):
        """Locate root cause."""

        traceback_text = runtime_error_info.get("traceback")
        crash_location = TracebackParser().main(traceback_text)

        merge_detector = MergeMismatchDetector()
        if merge_detector.is_merge_mismatch(
                                    runtime_error_info=runtime_error_info,
                                    crash_location=crash_location,
                                    origin_code=origin_code,
                                    traced_shapes=traced_shapes or {},
                                    initial_input_shapes=initial_input_shapes or {},
                                    source_path=self.origin_pyfile
                                ):

            rules_path = "step_two/merge_shape_debug/api_rules.json"
            merge_locator = MergeMismatchRootCauseLocator(mismatch_rules=rules_path)
            location_result = merge_locator.locate(
                origin_code=origin_code,
                runtime_error_info=runtime_error_info,
                traced_shapes=traced_shapes or {},
                initial_input_shapes=initial_input_shapes or {},
                source_path=self.origin_pyfile,
            )

            if location_result is not None:
                return location_result

        try:
            mapping = TracebackToInitMapper().main(
                origin_code,
                traceback_text,
                runtime_error_info=runtime_error_info,
            )
            location_result = build_general_root_cause_result(
                mapping=mapping,
                source_code=origin_code,
            )

            if location_result is not None:
                return location_result

        except ValueError:
            forward_local_locator = ForwardLocalCallLocator()
            local_forward_call_root_cause = forward_local_locator.locate(
                runtime_error_info=runtime_error_info,
            )

            if local_forward_call_root_cause is not None:
                return local_forward_call_root_cause

            return build_unresolved_root_cause_result(crash_location)

    def should_annotate_shapes(self, root_cause: [Dict[str, Any]]):
        if root_cause is None:
            return True

        crash_location = root_cause.get("crash_location")
        if crash_location is None:
            return True

        crash_stage = crash_location.get("crash_stage", "unknown")
        is_init_stage_crash = crash_stage == "init"

        flag_should_annotate_shapes = not is_init_stage_crash

        return flag_should_annotate_shapes

    def annotate_shapes(
        self,
        origin_code: str,
        traced_shapes: Dict[int, Any],
        dll_type: str,
        initial_input_shapes: Dict[int, Any] | None = None,
        root_cause: Dict[str, Any] | None = None,
    ):
        """Annotate shapes."""
        cleaner = CommentCleaner(dll_type)
        cleaned_code = cleaner.remove_comments_in_target_method(origin_code)

        annotator = ShapeAnnotation()
        annotated_code = annotator.annotation(
            cleaned_code,
            traced_shapes,
            initial_input_shapes=initial_input_shapes,
            root_cause=root_cause,
        )

        return annotated_code

    def save_final_code(self, final_code: str):
        """Save final code."""
        if not os.path.exists(self.annotated_pyfile):
            os.makedirs(os.path.dirname(self.annotated_pyfile), exist_ok=True)
        with open(self.annotated_pyfile, "w", encoding="utf-8") as file:
            file.write(final_code)

    def save_step2_result(self, result: Dict[str, Any]):
        """Save step2 result."""
        base_path, _ = os.path.splitext(self.annotated_pyfile)
        result_json_path = f"{base_path}.json"
        os.makedirs(os.path.dirname(result_json_path), exist_ok=True)
        with open(result_json_path, "w", encoding="utf-8") as file:
            json.dump(compact_step2_result_for_save(result), file, ensure_ascii=False, indent=2)

    def main(self):
        """Run the main workflow."""

        with open(self.origin_pyfile, "r", encoding="utf-8") as file:
            origin_code = file.read()

        if origin_code is None or origin_code.strip() == "":
            logger.error("Origin code is empty.")
            return None

        else:
            traced_shapes = {}
            initial_input_shapes = {}
            try:
                tracer = TensorShapeTrace(python_code=origin_code, filename=self.origin_pyfile)
                trace_result = tracer.main()
                traced_shapes = trace_result["traced_shapes"]
                initial_input_shapes = trace_result.get("initial_input_shapes", {})
                runtime_error_info = trace_result["runtime_error_info"]

            except Exception as e:
                logger.error(f"TensorShapeTrace failed: {e}")
                runtime_error_info = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                    "stage": "tensor_shape_trace_failed"
                }

            if runtime_error_info is not None:
                runtime_error_info["traceback"] = remove_local_path_info(
                    runtime_error_info.get("traceback")
                )

            if runtime_error_info is None:
                self.save_final_code(origin_code)
                result = {
                    "runtime_error_info": runtime_error_info,
                    "origin_code": origin_code,
                    "final_code": origin_code
                }
                self.save_step2_result(result)
                return result
            
            else:
                located_root_cause = self.locate_root_cause(
                                                origin_code=origin_code, 
                                                runtime_error_info=runtime_error_info, 
                                                traced_shapes=traced_shapes,
                                                initial_input_shapes=initial_input_shapes,
                                            )

                if located_root_cause['root_cause_type'] == "Unresolved_Crash":
                    self.save_final_code(origin_code)
                    result = {
                        "runtime_error_info": runtime_error_info,
                        "root_cause": located_root_cause,
                        "origin_code": origin_code,
                        "final_code": origin_code
                    }
                    self.save_step2_result(result)
                    return result

                else:
                    masked_code = self.root_cause_api_masker.mask(
                                                    origin_code=origin_code,
                                                    root_cause=located_root_cause,
                                                )
                    
                    should_annotate = self.should_annotate_shapes(root_cause=located_root_cause)
                    if should_annotate:
                        annotated_code = self.annotate_shapes(
                                                        origin_code=masked_code,
                                                        traced_shapes=traced_shapes,
                                                        dll_type=self.dll_type,
                                                        initial_input_shapes=initial_input_shapes,
                                                        root_cause=located_root_cause,
                                                    )
                        final_code = annotated_code
                    else:
                        annotated_code = None
                        final_code = masked_code

                    self.save_final_code(final_code)
                    result = {
                        "traced_shapes": traced_shapes,
                        "initial_input_shapes": initial_input_shapes,
                        "runtime_error_info": runtime_error_info,
                        "root_cause": located_root_cause,
                        "shape_annotation": should_annotate,
                        "origin_code": origin_code,
                        "final_code": final_code
                    }

                    self.save_step2_result(result)

                    return result



def resolve_batch_paths(
    llm_name: str,
    dll: str,
    experiment_id: str,
    input_root: str | None = None,
    output_root: str | None = None,
    log_root: str | None = None,
) -> tuple[Path, Path, Path]:
    """Resolve source, post-processing, and log paths for one batch.

    Custom roots are stage roots containing model directories, which makes the
    same entry point usable for ablation outputs without changing the default
    main-experiment layout.
    """
    dll_key = dll.strip().lower()

    if input_root:
        source_dir = Path(input_root) / llm_name / dll_key
    else:
        source_dir = Path("response") / llm_name / "step_one" / dll_key

    if output_root:
        target_dir = Path(output_root) / llm_name / dll_key
    else:
        target_dir = Path("response") / llm_name / "step_two" / dll_key

    if log_root:
        log_path = Path(log_root) / llm_name / dll_key / f"{experiment_id}.log"
    else:
        log_path = Path("logs") / llm_name / "step_two" / dll_key / f"{experiment_id}.log"

    return source_dir, target_dir, log_path


def build_postprocess_output_path(
    source_file: str | Path,
    source_dir: str | Path,
    target_dir: str | Path,
) -> Path:
    """Map a source program to a separate post-processing tree."""
    relative_path = Path(source_file).resolve().relative_to(Path(source_dir).resolve())
    return Path(target_dir) / relative_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type = str, help="The Inference LLM id")
    parser.add_argument("--dlls", type=str, help="List of DLLs")
    parser.add_argument("--experiment_id", type = str, help="The experiment id")
    parser.add_argument(
        "--input_root",
        type=str,
        default=None,
        help=(
            "Optional input stage root containing model directories. "
            "Defaults to response/<model>/step_one."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help=(
            "Optional output stage root containing model directories. "
            "Defaults to response/<model>/step_two."
        ),
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default=None,
        help="Optional log root containing model directories.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip a program when its output sidecar JSON already exists.",
    )
    args = parser.parse_args()

    dlls_list = json.loads(args.dlls) if args.dlls else []

    def log_handler(result, annotated_pyfile):
        runtime_error_info = result["runtime_error_info"]
        if runtime_error_info is None:
            logger.info("[Runtime_Error_Info]: None")
        else:
            logger.info("[Runtime_Error_Info] detected:\n"
                        f"{runtime_error_info['traceback']}")
            root_cause = result["root_cause"]
            if root_cause['root_cause_type'] == "Unresolved_Crash":
                logger.info(
                    f"[Root_Cause]: unresolved at {root_cause['crash_location']['crash_stage']} stage, "
                    f"[crash_code]: {root_cause['crash_location']['crash_code']}"
                )
            else:
                if root_cause["crash_location"]["crash_stage"] == "init":
                    logger.info(f"[Root_Cause_Type]: {root_cause['root_cause_type']}")
                    logger.info(
                        f"[Root_Cause_Location]: Crash in the [{root_cause['crash_location']['crash_stage']}] stage at API call: {get_root_cause_original_call(root_cause)}."
                    )
                elif root_cause["crash_location"]["crash_stage"] == "forward":
                    logger.info(f"[Root_Cause_Type]: {root_cause['root_cause_type']}")
                    if root_cause['root_cause_type'] == "Tensor_Merge_Mismatch":
                        logger.info(f"This is a {root_cause['mismatch_types']} mismatch.")
                        for target_layer in root_cause['target_layers']:
                            logger.info(
                                f"Possible root-cause parameters in target layer "
                                f"[{target_layer['original_call'].split('#')[0]}]: "
                                f"{target_layer.get('params_to_mask', [])}"
                            )
                    logger.info(
                        f"[Root_Cause_Location]: Crash in the [{root_cause['crash_location']['crash_stage']}] stage at [__init__] API call: {get_root_cause_original_call(root_cause).split('#')[0]}."
                    )
                    logger.info("[Shape_Annotation]: "f"{result['shape_annotation']}")
                else:
                    logger.info(
                        f"[Root_Cause_Location]: Crash in the [{root_cause['crash_location']['crash_stage']}] stage."
                    )

            logger.info(f"[Annotated or Origin Pyfile]: {annotated_pyfile}")

    llm_name = get_llm_name(args.model_id)
    for dll in dlls_list:
        draft_code_path, postprocess_code_path, log_path = resolve_batch_paths(
            llm_name=llm_name,
            dll=dll,
            experiment_id=args.experiment_id,
            input_root=args.input_root,
            output_root=args.output_root,
            log_root=args.log_root,
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler_id = logger.add(str(log_path))

        try:
            if not draft_code_path.exists():
                logger.warning(f"Input directory does not exist, skipping: {draft_code_path}")
                continue

            pyfiles = get_all_files(str(draft_code_path), file_type=".py")

            for pyfile in pyfiles:
                logger.info("=" * 100)
                logger.info(f"Processing pyfile: {pyfile}")
                annotated_pyfile = build_postprocess_output_path(
                    source_file=pyfile,
                    source_dir=draft_code_path,
                    target_dir=postprocess_code_path,
                )
                result_json_path = annotated_pyfile.with_suffix(".json")
                if args.skip_existing and result_json_path.exists():
                    logger.info(f"Skipping existing result: {result_json_path}")
                    continue

                dll_type = dll.strip().lower()
                if dll_type not in TensorShapeDebugPipeline.SUPPORTED_DLL_TYPES:
                    logger.warning(f"Unsupported DLL type, skipping: {dll}")
                    continue

                experiment = TensorShapeDebugPipeline(
                                                origin_pyfile=pyfile,
                                                annotated_pyfile=str(annotated_pyfile),
                                                dll_type=dll_type
                                            )

                try:
                    result = experiment.main()
                    if result is None:
                        logger.error("Failed to debug tensor shape.")
                        continue
                except Exception as e:
                    logger.error(f"Failed to process pyfile {pyfile}: {e}")
                    logger.error(traceback.format_exc())
                    continue

                log_handler(result, annotated_pyfile)
        finally:
            logger.remove(handler_id)
