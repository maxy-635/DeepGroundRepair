import argparse
from loguru import logger
from ablations import common
from ablations.A2.shape_repair import ShapeOnlyRepairPipeline


ABLATION_ID = "A2"


def run_step3(model_id: str, dlls: list[str], experiment_id: str, cache_dir: str) -> None:
    """Run step3."""
    llm_name = common.model_name(model_id)
    sampling_config_sink_ids = common.configure_sampling_config_sinks(
        ABLATION_ID, "step_three", model_id, experiment_id, dlls
    )
    try:
        llm_runtime = common.build_llm_runtime(
            model_id=model_id,
            cache_dir=cache_dir,
            max_tokens=3000,
        )
    finally:
        common.remove_log_sinks(sampling_config_sink_ids)

    for dll in dlls:
        canonical_dll = common.normalize_dll(dll)
        handler_id = common.configure_logger(ABLATION_ID, model_id, experiment_id, "step_three", canonical_dll)
        try:
            logger.info(f"[{ABLATION_ID}/{llm_name}/{dll}/Step3] Starting code repair.")
            step2_root = common.step2_dir(ABLATION_ID, llm_name, canonical_dll, experiment_id)
            repair_root = common.repair_dir(ABLATION_ID, llm_name, canonical_dll, experiment_id)

            ShapeOnlyRepairPipeline(
                step2_result_path=step2_root,
                repair_code_save_path=repair_root,
                llm_runtime=llm_runtime,
            ).main()

            logger.info(f"[{ABLATION_ID}/{llm_name}/{dll}/Step3] All tasks completed.")
        finally:
            logger.remove(handler_id)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run A2 Step Three to repair shape-annotated code.")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--dlls", nargs="+", required=True)
    parser.add_argument("--experiment_id", type=str, required=True)
    args = parser.parse_args()

    run_step3(
        model_id=args.model_id,
        dlls=args.dlls,
        experiment_id=args.experiment_id,
        cache_dir=args.cache_dir,
    )
