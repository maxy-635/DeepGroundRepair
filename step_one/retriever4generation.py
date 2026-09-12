import re
import os
import json
import argparse
import traceback
import torch
from loguru import logger
from deepcode_generation import DLcodeGeneration
from step_one.api_validation import api_validation
from step_one.whoosh_search import WhooshSearch, JiebaAnalyzer
from step_one.vector_store_manager import VectorStoreManager
from step_one.prompts.first_prompt import FirstPromptDesigner
from step_one.prompts.second_prompt import SecondPromptDesigner
from utils.chat2gemma import Chat2GemmaLLM
from utils.chat2mistral import Chat2MistralLLM
from utils.utils import dict2json, get_all_files, get_sampling_temperature, get_llm_name, read_yaml_data
os.environ['TF_USE_LEGACY_KERAS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Retrieval4Generation:
    """Retrieve API documentation and generate draft code."""
    _model = None
    _tokenizer = None
    _model_loaded = False



    def __init__(self, whoosh_searcher, embedding_searcher, model_id, model, tokenizer, chat2llm, saver):
        self.whoosh_searcher = whoosh_searcher
        self.embedding_searcher = embedding_searcher
        self.model_id = model_id
        self.model = model
        self.tokenizer = tokenizer
        self.chat2llm = chat2llm
        self.saver = saver

    @classmethod
    def from_config(cls, config):
        """Create an instance from configuration."""
        whoosh = WhooshSearch(**config["whoosh"])
        embedder = VectorStoreManager(**config["embedding"])
        model_id = config['model_config']['model_id']
        model_key = model_id.lower()
        if "gemma" in model_key:
            chat2llm = Chat2GemmaLLM
        elif "mistral" in model_key:
            chat2llm = Chat2MistralLLM
        else:
            raise ValueError("The inference_model_id is invalid, please check!")

        if not cls._model_loaded:
            cls._model, cls._tokenizer = chat2llm().load_model(
                model_id, 
                config['model_config']['cache_dir'], 
                config['model_config']['data_dtype']
            )
            cls._model_loaded = True
            logger.info(f"Loading inference model {model_id}")

        return cls(
            whoosh_searcher = whoosh,
            embedding_searcher = embedder,
            model_id = model_id,
            model=cls._model,
            tokenizer=cls._tokenizer,
            chat2llm = chat2llm,
            saver = config["saver"]
        )

    @staticmethod
    def _extract_candidate_api_code(response):
        """Extract candidate api code."""
        def strip_markdown_fence_lines(text):
            return re.sub(
                r"^\s*```\s*(?:[\w.+-]+)?\s*$",
                "",
                text,
                flags=re.MULTILINE,
            ).strip()

        code_blocks = re.findall(r"```(?:[\w.+-]+)?\s*(.*?)```", response, re.DOTALL)
        for code_block in code_blocks:
            cleaned_code_block = strip_markdown_fence_lines(code_block)
            if "candidate_apis" in cleaned_code_block:
                return cleaned_code_block

        cleaned_response = strip_markdown_fence_lines(response)
        return cleaned_response if cleaned_response else response.strip()

    @staticmethod
    def _parse_candidate_apis(code_inside):
        """Parse candidate apis."""
        candidate_apis = []
        pattern1 = r"candidate_apis\s*=\s*\[(.*?)\]"
        pattern2 = r"candidate_apis\.append\((\{.*?\})\)"
        fallback_pattern = r"""['"]api_name['"]\s*:\s*(?:['"]api_name['"]\s*:\s*)?['"]([^'"]+)['"]"""

        if "candidate_apis" not in code_inside:
            return candidate_apis

        if re.search(pattern1, code_inside, re.DOTALL) or re.search(pattern2, code_inside, re.DOTALL):
            try:
                local_vars = {"candidate_apis": []}
                exec(code_inside, globals(), local_vars)
                candidate_apis = local_vars["candidate_apis"]
            except Exception as e:
                logger.error(f"An error occurred while executing code: {e}")

        if candidate_apis:
            return candidate_apis

        api_names = list(dict.fromkeys(re.findall(fallback_pattern, code_inside)))

        candidate_apis = [
            {"api_name": api_name}
            for api_name in api_names
        ]
        if candidate_apis:
            logger.info(f"Parsed {len(candidate_apis)} APIs from incomplete candidate_apis output.")
        return candidate_apis

    def api_recommendation(self, task, dll):
        dlcode_generator = DLcodeGeneration(count_num=None, save_code_path=None, save_log_path=None)
        prompt = FirstPromptDesigner().prompt(task, dll)
        response = dlcode_generator.prompt2llm(
            model = self.model,
            tokenizer = self.tokenizer,
            prompt = prompt,
            chat2llm = self.chat2llm(),
            temperature = get_sampling_temperature(self.model_id),
            top_p = 0.95,
            max_new_tokens = 500
        )

        logger.info(f"API recommendation prompt:{prompt}")
        logger.info(f"API recommendation result:\n{response}")

        code_inside = self._extract_candidate_api_code(response)
        candidate_apis = self._parse_candidate_apis(code_inside)

        try:
            candidate_api_names = {'existing': [], 'no_existing': [], 'ordered': []}
            seen_apis = set()
            if not candidate_apis:
                logger.info("candidate_apis is empty; skipping API retrieval.")
            else:
                for candidate_api in candidate_apis:
                    api_name = candidate_api['api_name']

                    def api_prefix_append(api_name, dll):
                        dll_name = dll.strip().lower()
                        if dll_name == 'tensorflow':
                            if api_name.startswith('tf.'):
                                return api_name.replace('tf.', 'tensorflow.', 1)
                            if not api_name.startswith('tensorflow.'):
                                return f'tensorflow.{api_name}'
                        elif dll_name == 'pytorch':
                            if not api_name.startswith('torch.'):
                                return f'torch.{api_name}'
                        elif dll_name == 'paddlepaddle':
                            if not api_name.startswith('paddle.'):
                                return f'paddle.{api_name}'
                        return api_name
                    
                    candidate_api_name = api_prefix_append(api_name,dll)
                    if candidate_api_name not in seen_apis:
                        seen_apis.add(candidate_api_name)

                        if api_validation(candidate_api_name):
                            candidate_api_names['existing'].append(candidate_api_name)
                            candidate_api_names['ordered'].append({
                                "api_name": candidate_api_name,
                                "exists": True,
                            })
                        else:
                            candidate_api_names['no_existing'].append(candidate_api_name)
                            candidate_api_names['ordered'].append({
                                "api_name": candidate_api_name,
                                "exists": False,
                            })

        except:
            logger.info(f"An error occurred while executing code: {traceback.format_exc()}")

        return candidate_api_names


    def classification_retriever(self, candidate_api_names, top_k):

        with open("step_one/api_canonical_aliases.json", "r", encoding="utf-8") as f:
            API_CANONICAL_ALIASES = json.load(f)

        results = []
        for candidate_api in candidate_api_names['ordered']:
            api_name = candidate_api["api_name"]

            if candidate_api["exists"]:
                api_name = API_CANONICAL_ALIASES.get(api_name, api_name)

                logger.info(f"Retrieving existing API {api_name} with BM25")
                bm25_result = self.whoosh_searcher.main(query=api_name, limit=1)
                if bm25_result:
                    logger.info(f"BM25 retrieved API: {bm25_result[0]['api_name']}")
                    results.append(bm25_result[0])
                else:
                    logger.info("bm25_result is empty")
            else:
                logger.info(f"Retrieving unresolved API {api_name} with embeddings")
                embedding_result = self.embedding_searcher.main(query=api_name, k=1)
                if embedding_result:
                    logger.info(f"Embedding retrieval returned API: {embedding_result[0]}")
                    embedding_bm25_result = self.whoosh_searcher.main(query=embedding_result[0].replace('*API Name*: ', ''), limit=1)
                    if embedding_bm25_result:
                        logger.info(f"BM25 over embedding results returned API: {embedding_bm25_result[0]['api_name']}")
                        results.append(embedding_bm25_result[0])
                    else:
                        logger.info("BM25 over embedding results returned no matches")
                else:
                    logger.info("embedding_results is empty")


        api_docs = []
        retrieved_api_names = []
        for result in results:
            retrieved_api_name = result['api_name'].replace('*API Name*: ', '').strip()
            if retrieved_api_name in retrieved_api_names:
                continue
            retrieved_api_names.append(retrieved_api_name)

            api_doc = (
                f"{result['api_signature']}\n"
                f"{result['api_description']}\n"
            )
            api_docs.append("\n")
            api_docs.append(api_doc)

            if len(retrieved_api_names) >= top_k:
                break

        return api_docs, retrieved_api_names


    def generate_draft_code(self, task, api_docs, dll, save_code_full_path, count):
        prompt = SecondPromptDesigner().prompt(task, api_docs, dll)
        save_log_full_path = (
            save_code_full_path
                .replace("response", "chat_logs", 1)
                .replace(self.saver["save_code_last_path"], self.saver["save_log_last_path"], 1)
        )
        dlcode_generator = DLcodeGeneration(
            count_num = count,
            save_code_path = save_code_full_path,
            save_log_path = save_log_full_path,
        )
        response = dlcode_generator.prompt2llm(
            model = self.model,
            tokenizer = self.tokenizer,
            prompt = prompt,
            chat2llm = self.chat2llm(),
            temperature = get_sampling_temperature(self.model_id),
            top_p = 0.95,
            max_new_tokens = 2000,
        )
        logger.info(f"Draft-generation prompt:\n {prompt}")
        logger.info(f"Draft-generation result:\n {response}")

        dlcode_generator.save(prompt, response)

        return response


    def multi_draftcode_generation(self, benchmark_path, model_id, step_id, dll, repeats_count, top_k):
        yaml_files = get_all_files(directory=benchmark_path, file_type=".yaml")
        llm_name = get_llm_name(model_id)

        retrieval_records = []
        retrieval_record_path = (
            f"./evaluation/retrieval_accuracy/retreived_results/{llm_name}/{step_id}/{dll.lower()}/"
            "retrieval_api_list.json"
        )
        os.makedirs(os.path.dirname(retrieval_record_path), exist_ok=True)

        for yaml_file in yaml_files:
            task_id = os.path.basename(yaml_file)[:-5]
            logger.info(f"Start processing task: {task_id}")
            task= read_yaml_data(yaml_file)
            task_requirement = task['Requirement']
            save_code_full_path = f"./response/{llm_name}/{step_id}/{dll.lower()}/{task_id}"

            for count in range(1, repeats_count + 1):
                candidate_apis = []
                candidate_api_names = self.api_recommendation(task = task_requirement, dll = dll)
                # step 1.2 hybrid retriever
                api_docs, retrieved_api_names = self.classification_retriever(candidate_api_names=candidate_api_names, top_k=top_k)

                retrieval_records.append({
                    "task_id": f"{task_id}",
                    "candidate_apis": {
                        candidate_api["api_name"]: candidate_api["exists"]
                        for candidate_api in candidate_api_names.get("ordered", [])
                    },
                    "retrieved_apis": retrieved_api_names,
                })
                dict2json(retrieval_records, retrieval_record_path)

                # step 1.3 generate draft code
                self.generate_draft_code(
                    task = task_requirement,
                    dll = dll,
                    api_docs = api_docs,
                    save_code_full_path = save_code_full_path,
                    count = count
                )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type = str, help="The Inference LLM id")
    parser.add_argument("--benchmark", type = str, help="The benchmark dataset path")
    parser.add_argument("--step_id", type = str, help="The step id")
    parser.add_argument("--dlls", type=str, help="List of DLLs")
    parser.add_argument("--repeats", type = int, help="The repeat times")
    parser.add_argument("--experiment_id", type = str, help="The experiment id")

    args = parser.parse_args()

    dlls_list = json.loads(args.dlls) if args.dlls else []
    llm_name = get_llm_name(args.model_id)
    for dll in dlls_list:
        file_sink_id = logger.add(f"logs/{llm_name}/{args.step_id}/{dll.lower()}/{args.experiment_id}.log")
        logger.info("Starting deep-learning code generation")
        logger.info(
            f"Sampling configuration: model_id={args.model_id}, temperature={get_sampling_temperature(args.model_id)}"
        )
        logger.info(f"Target DL library: {dll}")

        try:
            config = {
                "whoosh": {"docs_path": f"./api_parser/{dll.lower()}/apis_parsed_results", 
                           "index_dir": f"./database/whoosh/whoosh_{dll.lower()}"},
                "embedding": {"embedding_model": "models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181",
                            "cache_dir": "/home/bingxing2/home/scx8amp/huggingface/hub",
                            "store_path": f"./database/embedding/embedding_{dll.lower()}.faiss",
                            "doc_folder": f"./api_parser/{dll.lower()}/apis_parsed_results"},
                "model_config": {"model_id": args.model_id,
                                "cache_dir": "/home/bingxing2/home/scx8amp/huggingface/hub",
                                "data_dtype": torch.bfloat16},
                "saver": {"save_code_last_path": args.experiment_id,
                         "save_log_last_path": "logs_" + args.experiment_id}
            }

            agent = Retrieval4Generation.from_config(config=config)
            agent.multi_draftcode_generation(
                benchmark_path = args.benchmark,
                model_id = args.model_id,
                step_id = args.step_id,
                dll = dll,
                repeats_count = args.repeats,
                top_k = 10
            )
            logger.info(f"{dll} deep-learning code generation tasks completed")
        finally:
            logger.remove(file_sink_id)
