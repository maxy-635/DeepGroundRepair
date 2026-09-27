import argparse

from loguru import logger

from ablations import common
from baselines.deepcode_generation import DLcodeGeneration
from baselines.modules.zeroshot_prompt import ZeroshotPromptDesigner
from utils.utils import read_yaml_data


ABLATION_ID = "A1"


def run_step1(
    model_id: str,
    benchmark_path: str,
    dlls: list[str],
    repeats_count: int,
    experiment_id: str,
    cache_dir: str | None = None,
) -> None:
    """Run step1."""

    llm_name = common.model_name(model_id)
    sampling_config_sink_ids = common.configure_sampling_config_sinks(
        ABLATION_ID, "step_one", model_id, experiment_id, dlls
    )
    try:
        llm_runtime = common.build_llm_runtime(
            model_id=model_id,
            cache_dir=cache_dir,
            max_tokens=2000,
        )
    finally:
        common.remove_log_sinks(sampling_config_sink_ids)

    yaml_files = common.iter_benchmark_tasks(benchmark_path)

    for dll in dlls:
        canonical_dll = common.normalize_dll(dll)
        handler_id = common.configure_logger(
            ABLATION_ID, model_id, experiment_id, "step_one", canonical_dll
        )
        try:
            logger.info(
                f"[{ABLATION_ID}/{llm_name}/{canonical_dll}/Step1-ZS] Starting zero-shot draft generation."
            )

            for yaml_file in yaml_files:
                task_id = common.task_id_from_yaml(yaml_file)
                task = read_yaml_data(yaml_file)
                task_requirement = task["Requirement"]

                for count in range(1, repeats_count + 1):
                    logger.info(
                        f"[A1/Step1-ZS] Processing {canonical_dll}/{task_id}, repeat={count}"
                    )

                    prompt = ZeroshotPromptDesigner().prompt(task_requirement, dll)

                    response = llm_runtime.chat(prompt)

                    generator = DLcodeGeneration(
                        count_num=count,
                        save_code_path=str(
                            common.code_dir(ABLATION_ID, "step_one", llm_name, canonical_dll, experiment_id, task_id)
                        ),
                        save_log_path=str(
                            common.log_dir(ABLATION_ID, "step_one", llm_name, canonical_dll, experiment_id, task_id)
                        ),
                    )
                    logger.info(f"[A1/Zero-shot Draft] Prompt:\n{prompt}")
                    logger.info(f"[A1/Zero-shot Draft] Model response:\n{response}")
                    generator.save(prompt, response)

            logger.info(
                f"[{ABLATION_ID}/{llm_name}/{canonical_dll}/Step1-ZS] Zero-shot draft generation completed."
            )
        finally:
            logger.remove(handler_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run A1 Step One to generate zero-shot draft code.")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--dlls", nargs="+", required=True)
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--experiment_id", type=str, required=True)
    args = parser.parse_args()

    run_step1(
        model_id=args.model_id,
        benchmark_path=args.benchmark,
        dlls=args.dlls,
        repeats_count=args.repeats,
        experiment_id=args.experiment_id,
        cache_dir=args.cache_dir,
    )
