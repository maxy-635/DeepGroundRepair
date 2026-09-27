import os
import json
import argparse
from loguru import logger
from baselines.deepcode_generation import DLcodeGeneration
from baselines.modules.experiment_utils import code_dir, load_local_llm, log_dir
from baselines.modules.hybrid_rag import HybridRAGRetriever
from baselines.modules.hybridrag_prompt import HybridRAGPromptDesigner
from utils.utils import dict2json, get_all_files, get_sampling_temperature, get_llm_name, read_yaml_data
from step_one.whoosh_search import JiebaAnalyzer

BASELINE_ID = "B2"


class SingleTaskPipeline:
    """Implement the single task pipeline component."""

    def __init__(self, model_id: str, model_cache_path: str, embedding_model_dir: str, experiment_id: str, top_k: int) -> None:
        """Initialize the instance."""

        self.model_id = model_id
        self.experiment_id = experiment_id
        self.top_k = top_k
        self.llm_name = get_llm_name(model_id)
        self.model, self.tokenizer, self.chat2llm = load_local_llm(model_id, model_cache_path)
        self.embedding_model_dir = embedding_model_dir

    def run(self, task_requirement: str, task_id: str, dll: str, count: int) -> list[str]:
        """Run the operation."""
        dll_key = dll.lower()
        with logger.contextualize(dll=dll_key):
            retriever = HybridRAGRetriever(
                embedding_model_dir=self.embedding_model_dir,
                dll=dll,
                top_k=self.top_k,
            )
            retrieved_docs = retriever.retrieve(task_requirement)
            retrieved_apis = {
                doc.get("api_name", "").replace("*API Name*: ", "").strip(): doc.get("fusion_score", doc.get("score", 0))
                for doc in retrieved_docs if doc.get("api_name", "")
            }

            prompt = HybridRAGPromptDesigner().prompt(
                task_requirement, dll, retrieved_docs
            )
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

            return retrieved_apis



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
            retrieval_records = []
            retrieval_record_path = (
                f"./baselines/evaluation/retrieval_accuracy/retreived_results/"
                f"{self.single_task_pipeline.llm_name}/{BASELINE_ID}/{dll_key}/"
                "retrieval_api_list.json"
            )
            os.makedirs(os.path.dirname(retrieval_record_path), exist_ok=True)

            for yaml_file in benchmark_files:
                task_id = os.path.basename(yaml_file)[:-5]
                task = read_yaml_data(yaml_file)
                task_requirement = task["Requirement"]
                for count in range(1, repeats_count + 1):
                    dll_logger.info(f"Processing:{yaml_file}")
                    retrieved_apis = self.single_task_pipeline.run(task_requirement, task_id, dll, count)

                    retrieval_records.append({
                        "task_id": f"{task_id}",
                        "retrieved_apis": retrieved_apis,
                    })
                    dict2json(retrieval_records, retrieval_record_path)

            dll_logger.info(f"[{BASELINE_ID}/{self.single_task_pipeline.llm_name}] Finished processing {dll}.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--model_cache_path", type=str, required=True)
    parser.add_argument("--embedding_model_dir", type=str, required=True)
    parser.add_argument("--benchmark", type=str, required=True)
    parser.add_argument("--dlls", type=str, help="List of DLLs")
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--experiment_id", type=str, required=True)
    args = parser.parse_args()

    logger.info(
        f"Sampling configuration: model_id={args.model_id}, temperature={get_sampling_temperature(args.model_id)}"
    )
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
        embedding_model_dir=args.embedding_model_dir,
        experiment_id=args.experiment_id,
        top_k=10,
    )
    MultiDLcodeGeneration(single).run(args.benchmark, dlls_list, args.repeats)
