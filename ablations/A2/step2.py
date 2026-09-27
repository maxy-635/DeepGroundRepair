import argparse
from pathlib import Path
from loguru import logger
from ablations import common
from ablations.A2.shape_annotation import ShapeAnnotationOnlyPipeline


ABLATION_ID = "A2"


def resolve_draft_root(draft_root: str, llm_name: str, dll: str, experiment_id: str) -> Path:
    """Resolve draft root."""
    if draft_root:
        return Path(draft_root) / dll
    return common.code_dir(ABLATION_ID, "step_one", llm_name, dll, experiment_id)

def run_step2(model_id: str, dlls: list[str], experiment_id: str, draft_root: str) -> None:
    """Run step2."""
    llm_name = common.model_name(model_id)

    for dll in dlls:
        canonical_dll = common.normalize_dll(dll)
        handler_id = common.configure_logger(ABLATION_ID, model_id, experiment_id, "step_two", canonical_dll)
        try:
            logger.info(f"[{ABLATION_ID}/{llm_name}/{dll}/Step2] Starting draft processing.")
            source_root = resolve_draft_root(
                draft_root=draft_root,
                llm_name=llm_name,
                dll=canonical_dll,
                experiment_id=experiment_id,
            )
            step2_root = common.step2_dir(ABLATION_ID, llm_name, canonical_dll, experiment_id)
            ShapeAnnotationOnlyPipeline.run_code_dir(
                source_code_root=source_root,
                target_step2_root=step2_root,
                dll=canonical_dll,
            )
            logger.info(f"[{ABLATION_ID}/{llm_name}/{dll}/Step2] All tasks completed.")
        finally:
            logger.remove(handler_id)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run A2 Step Two to generate shape-annotated code and diagnostic JSON only.")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--dlls", nargs="+", required=True)
    parser.add_argument("--experiment_id", type=str, required=True)
    parser.add_argument("--draft_root", type=str, required=True, help="Main-experiment Step One output root")


    args = parser.parse_args()
    run_step2(
        model_id=args.model_id,
        dlls=args.dlls,
        experiment_id=args.experiment_id,
        draft_root=args.draft_root,
    )
