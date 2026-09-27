import argparse
import json
from loguru import logger
from pathlib import Path
from ablations import common
from baselines.modules.debugging import Validation
from step_two.tensor_shape_debug import TensorShapeDebugPipeline
from step_two.utils import compact_step2_result_for_save

ABLATION_ID = "A1"


def run_step2(
    model_id: str,
    dlls: list[str],
    experiment_id: str,
) -> None:
    """Run step2."""
    llm_name = common.model_name(model_id)
    validator = Validation()

    for dll in dlls:
        canonical_dll = common.normalize_dll(dll)
        handler_id = common.configure_logger(ABLATION_ID, model_id, experiment_id, "step_two", canonical_dll)
        try:
            logger.info(f"[{ABLATION_ID}/{llm_name}/{canonical_dll}/Step2] Starting draft processing.")
            draft_root = common.code_dir(ABLATION_ID, "step_one", llm_name, canonical_dll, experiment_id)
            step2_root = common.step2_dir(ABLATION_ID, llm_name, canonical_dll, experiment_id)

            source_root = Path(draft_root)
            target_root = Path(step2_root)

            for pyfile in common.iter_pyfiles(source_root):
                annotated_pyfile = common.annotated_path_for_pyfile(pyfile, source_root, target_root)
                result_json = annotated_pyfile.with_suffix(".json")
                if result_json.exists():
                    logger.info(f"[Step2] Skipping existing result: {result_json}")
                    continue

                logger.info(f"[Step2] Processing: {pyfile}")
                verification = validator.compile_code_no_weight(str(pyfile))

                if verification["compile_status"] == "compile success":
                    origin_code = pyfile.read_text(encoding="utf-8")
                    annotated_pyfile.parent.mkdir(parents=True, exist_ok=True)
                    annotated_pyfile.write_text(origin_code, encoding="utf-8")

                    fast_path_result = {
                        "runtime_error_info": None,
                        "root_cause": {"root_cause_type": "No_Runtime_Error"},
                        "origin_code": origin_code,
                        "final_code": origin_code,
                    }
                    result_json.write_text(
                        json.dumps(
                            compact_step2_result_for_save(fast_path_result),
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    continue
    
                if verification["compile_status"] == "compile failed":
                    common.patch_no_weight_for_ablation_step2(canonical_dll)
                    pipeline = TensorShapeDebugPipeline(
                        origin_pyfile=str(pyfile),
                        annotated_pyfile=str(annotated_pyfile),
                        dll_type=canonical_dll,
                    )
                    result = pipeline.main()

                    if result is None:
                        logger.error(f"[Step2] Processing failed: {pyfile}")

            logger.info(f"[{ABLATION_ID}/{llm_name}/{canonical_dll}/Step2] All tasks completed.")
        finally:
            logger.remove(handler_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run A1 Step Two to generate annotated code and diagnostic JSON.")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--dlls", nargs="+", required=True)
    parser.add_argument("--experiment_id", type=str, required=True)
    args = parser.parse_args()

    run_step2(
        model_id=args.model_id,
        dlls=args.dlls,
        experiment_id=args.experiment_id
    )
