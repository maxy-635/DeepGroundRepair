from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


class CsvReader:
    """Implement the csv reader component."""

    @staticmethod
    def read(path: Path, required_fields: set[str]) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            fields = set(reader.fieldnames or [])
            missing = required_fields - fields
            if missing:
                raise ValueError(
                    f"{path} is missing required columns: {sorted(missing)}"
                )
            return list(reader)


class EvaluationDataset:
    """Implement the evaluation dataset component."""

    def __init__(self, rows: list[dict[str, str]]):
        if not rows:
            raise ValueError("The evaluation contains no samples.")
        self.rows = rows

    @classmethod
    def from_csvs(cls, gold_path: Path, prediction_path: Path) -> "EvaluationDataset":
        gold_rows = CsvReader.read(
            gold_path,
            {"sample_id", "library", "gold_api_line", "crash_is_root_api"},
        )
        prediction_rows = CsvReader.read(
            prediction_path,
            {"sample_id", "library", "pred_api_line"},
        )
        gold_map = cls._index_unique(gold_rows, "gold")
        prediction_map = cls._index_unique(prediction_rows, "prediction")
        if gold_map.keys() != prediction_map.keys():
            missing_predictions = sorted(gold_map.keys() - prediction_map.keys())
            missing_gold = sorted(prediction_map.keys() - gold_map.keys())
            raise ValueError(
                "Gold/prediction sample mismatch: "
                f"missing predictions={missing_predictions}, missing gold={missing_gold}"
            )

        merged = []
        for sample_id in sorted(gold_map):
            gold = gold_map[sample_id]
            prediction = prediction_map[sample_id]
            if gold.get("library") != prediction.get("library"):
                raise ValueError(f"Library mismatch for sample: {sample_id}")
            merged.append(
                {
                    **gold,
                    "pred_api_line": prediction.get("pred_api_line", ""),
                }
            )
        return cls(merged)

    @staticmethod
    def _index_unique(
        rows: list[dict[str, str]], source: str
    ) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for row in rows:
            sample_id = (row.get("sample_id") or "").strip()
            if not sample_id:
                raise ValueError(f"Missing sample_id in {source} CSV")
            if sample_id in result:
                raise ValueError(f"Duplicate sample_id in {source} CSV: {sample_id}")
            result[sample_id] = row
        return result

    @staticmethod
    def model_from_sample_id(sample_id: str) -> str:
        model = sample_id.split("/", maxsplit=1)[0].strip()
        if not model:
            raise ValueError(f"Cannot infer model from sample_id: {sample_id}")
        return model

    def by_model(self) -> dict[str, list[dict[str, str]]]:
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in self.rows:
            model = self.model_from_sample_id(row["sample_id"])
            grouped.setdefault(model, []).append(row)
        return {model: grouped[model] for model in sorted(grouped)}


class RootCauseEvaluator:
    """Implement the root cause evaluator component."""

    @staticmethod
    def evaluate(model: str, rows: list[dict[str, str]]) -> list[dict[str, object]]:
        grouped: dict[str, list[dict[str, str]]] = {}
        for row_number, row in enumerate(rows, start=2):
            gold_line = RootCauseEvaluator.parse_line_number(
                row.get("gold_api_line", "")
            )
            crash_is_root = row.get("crash_is_root_api", "").strip().lower()
            if gold_line is None or crash_is_root not in {"yes", "no"}:
                raise ValueError(
                    f"{model}: gold row {row_number} is incomplete: "
                    "gold_api_line must be a positive integer, and "
                    "crash_is_root_api must be yes or no."
                )
            library = row.get("library", "").strip() or "Unknown"
            grouped.setdefault(library, []).append(row)

        summaries = [
            RootCauseEvaluator.summarize(model, library, library_rows)
            for library, library_rows in sorted(grouped.items())
        ]
        summaries.append(RootCauseEvaluator.summarize(model, "Overall", rows))
        return summaries

    @staticmethod
    def parse_line_number(value: object) -> int | None:
        try:
            line_number = int(str(value).strip())
        except (AttributeError, TypeError, ValueError):
            return None
        return line_number if line_number > 0 else None

    @staticmethod
    def summarize(
        model: str, library: str, rows: list[dict[str, str]]
    ) -> dict[str, object]:
        total = len(rows)
        api_correct = 0
        crash_root_different = 0
        for row in rows:
            predicted_line = RootCauseEvaluator.parse_line_number(
                row.get("pred_api_line", "")
            )
            gold_line = RootCauseEvaluator.parse_line_number(row["gold_api_line"])
            if predicted_line is not None and predicted_line == gold_line:
                api_correct += 1
            if row["crash_is_root_api"].strip().lower() == "no":
                crash_root_different += 1

        return {
            "model": model,
            "library": library,
            "shape_errors": total,
            "api_correct": api_correct,
            "api_localization_accuracy": api_correct / total,
            "crash_root_different": crash_root_different,
            "crash_root_diff_rate": crash_root_different / total,
        }

    @staticmethod
    def write_reports(
        report_root: Path,
        model: str,
        summaries: list[dict[str, object]],
    ) -> tuple[Path, Path]:
        model_dir = report_root / model
        model_dir.mkdir(parents=True, exist_ok=True)
        json_path = model_dir / "root_cause_accuracy.json"
        csv_path = model_dir / "root_cause_accuracy.csv"
        json_path.write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        return json_path, csv_path

    @classmethod
    def run(
        cls,
        gold_path: Path,
        prediction_path: Path,
        report_root: Path,
    ) -> dict[str, list[dict[str, object]]]:
        dataset = EvaluationDataset.from_csvs(gold_path, prediction_path)
        results = {}
        for model, rows in dataset.by_model().items():
            summaries = cls.evaluate(model, rows)
            cls.write_reports(report_root, model, summaries)
            results[model] = summaries
        return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold_annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--report_root",
        type=Path,
        default=Path("evaluation/root_cause_location_accuracy/report"),
        help="Root directory; one subdirectory is created for each model.",
    )
    args = parser.parse_args()

    results = RootCauseEvaluator.run(
        args.gold_annotations,
        args.predictions,
        args.report_root,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
