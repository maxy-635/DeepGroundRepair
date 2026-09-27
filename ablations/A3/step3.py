import argparse
from pathlib import Path
from loguru import logger
from ablations import common
from ablations.A3.pipeline import NoShapeContextRepairPipeline


ABLATION_ID = "A3"


def resolve_step2_root(
    step2_result_root: str | None,
    llm_name: str,
    dll: str,
    experiment_id: str,
) -> Path:
    """Resolve step2 root."""
    if step2_result_root:
        return Path(step2_result_root) / dll
    return Path("response") / llm_name / "step_two" / dll


def run_step3(
    model_id: str,
    dlls: list[str],
    experiment_id: str,
    cache_dir: str | None = None,
    max_tokens: int = 3000,
    step2_result_root: str | None = None,
) -> None:
    """Run step3."""
    llm_name = common.model_name(model_id)
    sampling_config_sink_ids = common.configure_sampling_config_sinks(
        ABLATION_ID, "step_three", model_id, experiment_id, dlls
    )
    try:
        llm_runtime = common.build_llm_runtime(
            model_id=model_id,
            cache_dir=cache_dir,
            max_tokens=max_tokens,
        )
    finally:
        common.remove_log_sinks(sampling_config_sink_ids)

    for dll in dlls:
        result_dll = common.normalize_dll(dll)
        handler_id = common.configure_logger(ABLATION_ID, model_id, experiment_id, "step_three", result_dll)
        try:
            logger.info(f"[{ABLATION_ID}/{llm_name}/{result_dll}/Step3] Starting code repair.")

            step2_root = resolve_step2_root(step2_result_root, llm_name, result_dll, experiment_id)
            if not step2_root.exists():
                raise FileNotFoundError(
                    f"A3 requires the main-experiment Step Two result directory: {step2_root}\n"
                    f"If the results are elsewhere, set --step2_result_root to the Step Two root, "
                    f"for example: response/{llm_name}/step_two"
                )

            repair_root = common.repair_dir(ABLATION_ID, llm_name, result_dll, experiment_id)

            NoShapeContextRepairPipeline(
                step2_result_path=step2_root,
                repair_code_save_path=repair_root,
                llm_runtime=llm_runtime,
            ).main()
            logger.info(f"[{ABLATION_ID}/{llm_name}/{result_dll}/Step3] All tasks completed.")
        finally:
            logger.remove(handler_id)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run A3 using main-experiment Step Two results without shape context.")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=3000)
    parser.add_argument("--dlls", nargs="+", required=True)
    parser.add_argument("--experiment_id", type=str, required=True)
    parser.add_argument("--step2_result_root", type=str, default=None, help="Main-experiment Step Two result root")
    args = parser.parse_args()

    run_step3(
        model_id=args.model_id,
        dlls=args.dlls,
        experiment_id=args.experiment_id,
        cache_dir=args.cache_dir,
        max_tokens=args.max_tokens,
        step2_result_root=args.step2_result_root,
    )
