import os
import json
import argparse
from loguru import logger
from baselines.deepcode_generation import DLcodeGeneration
from baselines.modules.experiment_utils import code_dir, load_local_llm, log_dir
from baselines.modules.zeroshot_prompt import ZeroshotPromptDesigner
from utils.utils import get_all_files, get_sampling_temperature, get_llm_name, read_yaml_data


BASELINE_ID = "B1"


class SingleTaskPipeline:
    """Implement the single task pipeline component."""

    def __init__(self, model_id: str, model_cache_path: str, experiment_id: str) -> None:
        """Initialize the instance."""

        self.model_id = model_id
        self.experiment_id = experiment_id
        self.llm_name = get_llm_name(model_id)
        self.model, self.tokenizer, self.chat2llm = load_local_llm(model_id, model_cache_path)

    def run(self, task_requirement: str, task_id: str, dll: str, count: int) -> dict:
        """Run the operation."""
        dll_key = dll.lower()
        with logger.contextualize(dll=dll_key):
            prompt = ZeroshotPromptDesigner().prompt(task_requirement, dll)
            save_code_path = code_dir(self.llm_name, BASELINE_ID, dll_key, self.experiment_id, task_id)
            save_log_path = log_dir(self.llm_name, BASELINE_ID, dll_key, self.experiment_id, task_id)

            generator = DLcodeGeneration(
                count_num=count,
                save_code_path=save_code_path,
                save_log_path=save_log_path,
            )
            response = generator.prompt2llm(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                chat2llm=self.chat2llm,
                temperature=get_sampling_temperature(self.model_id),
                top_p=0.95,
                max_new_tokens=2000,
            )

            task_logger = logger.bind(dll=dll_key)
            task_logger.info(f"Prompt:\n{prompt}")
            task_logger.info(f"Response:\n{response}")

            generator.save(prompt, response)



class MultiDLcodeGeneration:
    """Implement the multi dlcode generation component."""

    def __init__(self, single_task_pipeline: SingleTaskPipeline) -> None:
        """Initialize the instance."""
        self.single_task_pipeline = single_task_pipeline

    def run(self, benchmark_path: str, dlls: list[str], repeats_count: int) -> None:
        """Run the operation."""
        benchmark_files = get_all_files(directory=benchmark_path, file_type=".yaml")

        for dll in dlls:
            dll_key = dll.lower()
            dll_logger = logger.bind(dll=dll_key)
            dll_logger.info(f"[{BASELINE_ID}/{self.single_task_pipeline.llm_name}] Start processing {dll}.")
            dll_logger.info(
                f"Sampling configuration: model_id={self.single_task_pipeline.model_id}, temperature={get_sampling_temperature(self.single_task_pipeline.model_id)}"
            )
            for yaml_file in benchmark_files:
                task_id = os.path.basename(yaml_file)[:-5]
                task = read_yaml_data(yaml_file)
                task_requirement = task["Requirement"]
                for count in range(1, repeats_count + 1):
                    dll_logger.info(f"Processing:{yaml_file}")
                    self.single_task_pipeline.run(task_requirement, task_id, dll, count)
            dll_logger.info(f"[{BASELINE_ID}/{self.single_task_pipeline.llm_name}] Finished processing {dll}.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--model_cache_path", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--dlls", type=str, help="List of DLLs")
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--experiment_id", type=str, required=True)
    args = parser.parse_args()

    dlls_list = json.loads(args.dlls)

    llm_name = get_llm_name(args.model_id)
    log_dir_path = f"./baselines/logs/{BASELINE_ID}/{llm_name}"
    os.makedirs(log_dir_path, exist_ok=True)
    for dll in dlls_list:
        dll_key = dll.lower()
        logger.add(
            f"{log_dir_path}/{dll_key}/{args.experiment_id}.log",
            filter=lambda record, dll_key=dll_key: record["extra"].get("dll") == dll_key
        )

    single = SingleTaskPipeline(
        model_id=args.model_id,
        model_cache_path=args.model_cache_path,
        experiment_id=args.experiment_id,
    )
    MultiDLcodeGeneration(single).run(args.benchmark, dlls_list, args.repeats)
