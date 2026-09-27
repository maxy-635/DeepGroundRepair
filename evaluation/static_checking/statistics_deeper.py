import json
import os
from collections import OrderedDict



class DeeperStatistics:

    def __init__(self, report_root, model_ids, step_id, dlls, experiment_id):
        self.report_root = report_root
        self.model_ids = model_ids
        self.step_id = step_id
        self.dlls = dlls
        self.experiment_id = experiment_id

    @staticmethod
    def parse_count(value):
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return int(value.split("(")[0])
        raise ValueError(f"Unsupported count value: {value}")

    def count_errors(self, error_counts):
        statistics = {
            "categories": OrderedDict(
                (category, {"total": 0, "issues": OrderedDict()})
                for category in ERROR_CATEGORIES
            ),
            "uncategorized": OrderedDict(),
        }

        for issue, raw_count in error_counts.items():
            count = self.parse_count(raw_count)
            matched_category = None

            for category, issues in ERROR_CATEGORIES.items():
                if issue in issues:
                    matched_category = category
                    break

            if matched_category is None:
                statistics["uncategorized"][issue] = count
                continue

            statistics["categories"][matched_category]["issues"][issue] = count
            statistics["categories"][matched_category]["total"] += count

        statistics["categories"] = OrderedDict(
            sorted(
                statistics["categories"].items(),
                key=lambda item: item[1]["total"],
                reverse=True
            )
        )
        statistics["uncategorized"] = OrderedDict(
            sorted(statistics["uncategorized"].items(), key=lambda item: item[1], reverse=True)
        )
        statistics["total_categorized_errors"] = sum(
            category_counts["total"]
            for category_counts in statistics["categories"].values()
        )
        statistics["total_uncategorized_errors"] = sum(statistics["uncategorized"].values())
        statistics["total_errors"] = (
            statistics["total_categorized_errors"] + statistics["total_uncategorized_errors"]
        )

        return statistics

    def main(self):
        for model_id in self.model_ids:
            summary_json_file = os.path.join(
                self.report_root,
                model_id,
                self.step_id,
                f"{model_id}_messages_statistics.json",
            )
            save_path = os.path.join(
                self.report_root,
                model_id,
                self.step_id,
                f"{model_id}_deeper_statistics.json",
            )

            if not os.path.exists(summary_json_file):
                continue

            with open(summary_json_file, "r", encoding="utf-8") as file:
                data = json.load(file)

            model_statistics = {
                "source_file": summary_json_file,
                "summary": self.count_errors(data.get("summary", {}).get("error", {})),
            }

            for dll in self.dlls:
                if dll not in data:
                    continue

                error_counts = data[dll].get(self.experiment_id, {}).get("error", {})
                model_statistics[dll] = {
                    self.experiment_id: self.count_errors(error_counts)
                }

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as json_file:
                json.dump(model_statistics, json_file, indent=2, ensure_ascii=False)



if __name__ == "__main__":
    REPORT_ROOT = "evaluation/static_checking/report"
    MODEL_ID_LIST = [
        "gemma_3_4b_it",
        "gemma_3_12b_it",
        "gemma_3_27b_it",
        "ministral_3_3b_instruct_2512",
        "ministral_3_14b_instruct_2512",
        "mistral_small_3_2_24b_instruct_2506"
    ]
    STEP_ID = "step_three"
    DLLS = ["tensorflow", "pytorch", "paddlepaddle"]
    EXPERIMENT_ID = "models_0701"

    ERROR_CATEGORIES = OrderedDict({
        "API Hallucination": {
            "no-member",
            "import-error",
            "no-name-in-module",
        },
        "API Parameters Misuse": {
            "unexpected-keyword-arg",
            "no-value-for-parameter",
            "too-many-function-args",
            "redundant-keyword-arg",
            # "not-callable",
            # "bad-super-call",
        },
    })
    deeper_statistics = DeeperStatistics(
        report_root=REPORT_ROOT,
        model_ids=MODEL_ID_LIST,
        step_id=STEP_ID,
        dlls=DLLS,
        experiment_id=EXPERIMENT_ID,
    )
    deeper_statistics.main()
