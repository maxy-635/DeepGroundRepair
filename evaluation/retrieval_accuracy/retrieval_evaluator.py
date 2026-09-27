import argparse
import json
from pathlib import Path


DEFAULT_API_ALIASES_PATH = Path(__file__).with_name("api_aliases.json")


class ApiNormalizer:
    """Implement the api normalizer component."""

    def __init__(self, alias_path: Path = DEFAULT_API_ALIASES_PATH) -> None:
        if not alias_path.exists():
            self.aliases = {}
            return
        with alias_path.open("r", encoding="utf-8") as file:
            self.aliases = json.load(file)
        if not isinstance(self.aliases, dict):
            raise ValueError(f"api aliases must be a JSON object: {alias_path}")

    def normalize_list(self, api_names: list[str]) -> list[str]:
        normalized = []
        seen = set()
        for api_name in api_names:
            normalized_name = self.aliases.get(api_name, api_name)
            if normalized_name not in seen:
                normalized.append(normalized_name)
                seen.add(normalized_name)
        return normalized


class RetrievalMetrics:
    """Implement the retrieval metrics component."""

    METRICS = ("precision", "recall", "hit")

    @staticmethod
    def score_at_k(retrieved_items: list[str], gold_items: list[str], k: int) -> dict[str, float | int]:
        top_k = retrieved_items[:k]
        gold_set = set(gold_items)
        hit_count = sum(item in gold_set for item in top_k)
        return {
            "precision": hit_count / len(top_k) if top_k else 0,
            "recall": hit_count / len(gold_set) if gold_set else 0,
            "hit": int(hit_count > 0),
        }

    def summarize(self, records: list[dict], ks: list[int]) -> dict:
        per_sample = [{"sample_id": record["sample_id"]} for record in records]
        summary = {}

        for k in ks:
            scores = [
                self.score_at_k(record["retrieved_items"], record["gold_items"], k)
                for record in records
            ]
            for sample_result, score in zip(per_sample, scores):
                for metric in self.METRICS:
                    sample_result[f"{metric}@{k}"] = score[metric]

            summary[f"@{k}"] = {
                metric: sum(score[metric] for score in scores) / len(scores) if scores else 0
                for metric in self.METRICS
            }

        return {
            "total_samples": len(records),
            "ks": ks,
            "summary": summary,
            "per_sample": per_sample,
        }


class RetrievalEvaluator:
    """Implement the retrieval evaluator component."""

    def __init__(self, normalizer: ApiNormalizer, ks: list[int]) -> None:
        self.normalizer = normalizer
        self.ks = ks
        self.metrics = RetrievalMetrics()

    def load_records(self, input_path: Path, retrieval_path: Path | None = None) -> list[dict]:
        with input_path.open("r", encoding="utf-8") as file:
            input_records = json.load(file)
        if retrieval_path is None:
            records = input_records
        else:
            gold_by_task = {record["task_id"]: record.get("gold_apis", []) for record in input_records}
            records = []
            missing_gold = []

            with retrieval_path.open("r", encoding="utf-8") as file:
                retrieval_records = json.load(file)

            for record in retrieval_records:
                task_id = record.get("task_id")
                if task_id not in gold_by_task:
                    missing_gold.append(task_id)
                    continue
                records.append({
                    "sample_id": task_id,
                    "retrieved_items": record.get("retrieved_apis", []),
                    "gold_items": gold_by_task[task_id],
                })

            if missing_gold:
                preview = ", ".join(str(task_id) for task_id in missing_gold[:10])
                raise ValueError(f"missing gold set for {len(missing_gold)} task(s): {preview}")

        return [
            {
                "sample_id": record.get("sample_id"),
                "retrieved_items": self.normalizer.normalize_list(record.get("retrieved_items", [])),
                "gold_items": self.normalizer.normalize_list(record.get("gold_items", [])),
            }
            for record in records
        ]

    def evaluate(self, input_path: Path, output_path: Path, retrieval_path: Path | None = None) -> dict:
        result = self.metrics.summarize(self.load_records(input_path, retrieval_path), self.ks)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        return result

    def run_batch(
        self,
        model_ids: list[str],
        step_ids: list[str],
        dlls: list[str],
        experiment_id: str | None = None,
    ) -> None:
        for model_id in model_ids:
            for step_id in step_ids:
                for dll in dlls:
                    dll_key = dll.lower()
                    result = self.evaluate(
                        input_path=Path(f"evaluation/retrieval_accuracy/api_gold_sets/{dll_key}.json"),
                        retrieval_path=Path(
                            f"evaluation/retrieval_accuracy/retreived_results/"
                            f"{model_id}/{step_id}/{dll_key}/retrieval_api_list.json"
                        ),
                        output_path=Path(
                            f"evaluation/retrieval_accuracy/report/"
                            f"{model_id}/{step_id}/{dll_key}/retrieval_accuracy.json"
                        ),
                    )
                    print(f"{model_id}/{step_id}/{dll_key}/{experiment_id}")
                    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default=None, help="path to evaluator input json or api gold set json")
    parser.add_argument("--retrieval_json", type=str, default=None, help="path to retrieval_api_list.json")
    parser.add_argument("--output_json", type=str, default=None, help="path to save result json")
    parser.add_argument("--model_ids", type=str, default=None, help="JSON list of model ids")
    parser.add_argument("--step_ids", type=str, default=None, help="JSON list of step ids")
    parser.add_argument("--dlls", type=str, default=None, help="JSON list of DL framework names")
    parser.add_argument("--experiment_id", type=str, default=None, help="experiment id, e.g. models_0701")
    parser.add_argument("--api_aliases_json", type=str, default=str(DEFAULT_API_ALIASES_PATH))
    parser.add_argument("--ks", type=str, default="[10]", help="JSON list of K values")
    args = parser.parse_args()

    evaluator = RetrievalEvaluator(
        normalizer=ApiNormalizer(Path(args.api_aliases_json)),
        ks=json.loads(args.ks),
    )

    if args.step_ids:
        if not args.model_ids or not args.dlls:
            raise ValueError("--model_ids and --dlls are required in batch mode")
        evaluator.run_batch(
            model_ids=json.loads(args.model_ids),
            step_ids=json.loads(args.step_ids),
            dlls=json.loads(args.dlls),
            experiment_id=args.experiment_id,
        )
    else:
        if not args.input_json or not args.output_json:
            raise ValueError("--input_json and --output_json are required outside batch mode")
        result = evaluator.evaluate(
            input_path=Path(args.input_json),
            retrieval_path=Path(args.retrieval_json) if args.retrieval_json else None,
            output_path=Path(args.output_json),
        )
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
