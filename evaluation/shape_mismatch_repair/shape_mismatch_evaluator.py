import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from evaluation.shape_mismatch_repair.error_taxonomy import (
        classify_shape_mismatch,
        normalize_error_text,
    )
except ModuleNotFoundError:
    # Support direct execution via `python evaluation/.../shape_mismatch_evaluator.py`.
    from error_taxonomy import classify_shape_mismatch
    from error_taxonomy import normalize_error_text


class SampleIndex:
    """Implement the sample index component."""

    def __init__(self, records: dict[str, dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    @classmethod
    def from_sidecar_directory(cls, json_root: Path | str) -> "SampleIndex":
        json_root = Path(json_root)
        if not json_root.is_dir():
            raise FileNotFoundError(f"Sidecar JSON directory not found: {json_root}")

        records: dict[str, dict[str, Any]] = {}
        duplicates: list[str] = []
        for json_path in sorted(json_root.rglob("*.json")):
            with json_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            if not isinstance(data, dict):
                raise ValueError(f"Sidecar JSON must be an object: {json_path}")

            sample_id = cls.sample_key(str(json_path))
            if sample_id in records:
                duplicates.append(sample_id)
                continue
            records[sample_id] = {
                "json_path": str(json_path),
                "data": data,
            }

        cls._raise_on_duplicates(duplicates, json_root)
        return cls(records)

    @staticmethod
    def sample_key(file_path: str) -> str:
        path = Path(file_path)
        task_id = path.parent.name
        stem = path.stem
        for suffix in ("_annotated", "_repaired"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        return f"{task_id}/{stem}"

    @staticmethod
    def traceback(record: dict[str, Any]) -> str:
        runtime_error_info = record.get("runtime_error_info")
        if not isinstance(runtime_error_info, dict):
            return ""
        return normalize_error_text(runtime_error_info.get("traceback"))

    @staticmethod
    def root_cause_type(record: dict[str, Any]) -> str | None:
        top_level_type = record.get("root_cause_type")
        if top_level_type:
            return str(top_level_type)

        root_cause = record.get("root_cause")
        if isinstance(root_cause, dict) and root_cause.get("root_cause_type"):
            return str(root_cause["root_cause_type"])
        return None

    @staticmethod
    def _raise_on_duplicates(duplicates: list[str], source_path: Path) -> None:
        if duplicates:
            duplicate_text = ", ".join(sorted(set(duplicates)))
            raise ValueError(f"Duplicate sample keys found in {source_path}: {duplicate_text}")


@dataclass
class RepairSummary:
    """Implement the repair summary component."""

    initial_root_cause_type: str | None = None
    initial_shape_mismatch_cases: int = 0
    repaired_successfully: int = 0
    missing_final_record: int = 0
    final_root_cause_type_counts: dict[str, int] = field(default_factory=dict)

    def record(self, final_root_cause_type: str | None, repaired: bool) -> None:
        self.initial_shape_mismatch_cases += 1
        if repaired:
            self.repaired_successfully += 1
        if final_root_cause_type is None:
            self.missing_final_record += 1

        final_type_key = final_root_cause_type or "Missing_Final_Record"
        self.final_root_cause_type_counts[final_type_key] = (
            self.final_root_cause_type_counts.get(final_type_key, 0) + 1
        )

    def as_dict(self) -> dict[str, Any]:
        initial_cases = self.initial_shape_mismatch_cases
        result: dict[str, Any] = {
            "initial_shape_mismatch_cases": initial_cases,
            "repaired_successfully": self.repaired_successfully,
            "missing_final_record": self.missing_final_record,
            "final_root_cause_type_counts": dict(
                sorted(self.final_root_cause_type_counts.items())
            ),
            "shape_mismatch_repair_success_rate": (
                self.repaired_successfully / initial_cases if initial_cases else 0
            ),
        }
        if self.initial_root_cause_type is not None:
            result = {
                "initial_root_cause_type": self.initial_root_cause_type,
                **result,
            }
        return result


class ShapeMismatchRepairEvaluator:
    """Implement the shape mismatch repair evaluator component."""

    ROOT_CAUSE_TO_SCOPE = {
        "General_Crash": "intra_path",
        "Tensor_Merge_Mismatch": "inter_path",
    }

    def summarize_root_cause_pair(
        self,
        initial_json_root: Path | str,
        final_json_root: Path | str,
    ) -> dict[str, Any]:
        initial_json_root = Path(initial_json_root)
        final_json_root = Path(final_json_root)
        initial_index = SampleIndex.from_sidecar_directory(initial_json_root)
        final_index = SampleIndex.from_sidecar_directory(final_json_root)

        all_shape_summary = RepairSummary()
        classified_shape_summary = RepairSummary()
        scope_summaries = {
            scope: RepairSummary(initial_root_cause_type=root_cause_type)
            for root_cause_type, scope in self.ROOT_CAUSE_TO_SCOPE.items()
        }
        excluded_root_cause_types: dict[str, int] = {}
        per_sample: list[dict[str, Any]] = []

        for sample_id in sorted(initial_index.records):
            initial_entry = initial_index.records[sample_id]
            initial_data = initial_entry["data"]
            final_entry = final_index.records.get(sample_id)
            final_data = None if final_entry is None else final_entry["data"]

            is_shape_mismatch, matched_rule = classify_shape_mismatch(
                SampleIndex.traceback(initial_data)
            )
            if not is_shape_mismatch:
                continue

            initial_root_cause_type = SampleIndex.root_cause_type(initial_data)
            mismatch_scope = self.ROOT_CAUSE_TO_SCOPE.get(initial_root_cause_type)
            final_root_cause_type = (
                None if final_data is None else SampleIndex.root_cause_type(final_data)
            )
            repaired = final_root_cause_type == "No_Runtime_Error"

            all_shape_summary.record(final_root_cause_type, repaired)
            included_in_scope_analysis = mismatch_scope is not None
            if included_in_scope_analysis:
                classified_shape_summary.record(final_root_cause_type, repaired)
                scope_summaries[mismatch_scope].record(final_root_cause_type, repaired)
            else:
                excluded_key = initial_root_cause_type or "Missing_Root_Cause_Type"
                excluded_root_cause_types[excluded_key] = (
                    excluded_root_cause_types.get(excluded_key, 0) + 1
                )

            per_sample.append(
                {
                    "sample_id": sample_id,
                    "initial_json_path": initial_entry["json_path"],
                    "final_json_path": (
                        None if final_entry is None else final_entry["json_path"]
                    ),
                    "initial_is_shape_mismatch": True,
                    "initial_shape_rule": matched_rule,
                    "initial_root_cause_type": initial_root_cause_type,
                    "mismatch_scope": mismatch_scope,
                    "included_in_scope_analysis": included_in_scope_analysis,
                    "final_root_cause_type": final_root_cause_type,
                    "repaired_successfully": repaired,
                }
            )

        return {
            "evaluation_mode": "root_cause_sidecar_json",
            "initial_json_root": str(initial_json_root),
            "final_json_root": str(final_json_root),
            "classification": {
                "shape_mismatch_source": "initial_runtime_traceback_error_taxonomy",
                "mismatch_scope_source": "initial_root_cause_type",
                "root_cause_to_scope": self.ROOT_CAUSE_TO_SCOPE,
                "repair_success_condition": (
                    "final_root_cause_type == No_Runtime_Error"
                ),
            },
            "summary": {
                "all_initial_samples": len(initial_index),
                "all_final_samples": len(final_index),
                "all_shape_mismatches": all_shape_summary.as_dict(),
                "classified_shape_mismatches": classified_shape_summary.as_dict(),
                "by_mismatch_scope": {
                    scope: summary.as_dict()
                    for scope, summary in scope_summaries.items()
                },
                "excluded_shape_mismatches_by_root_cause_type": dict(
                    sorted(excluded_root_cause_types.items())
                ),
            },
            "per_sample": per_sample,
        }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial_json_root",
        type=str,
        help="Step 2 sidecar JSON directory for one model/library batch",
    )
    parser.add_argument(
        "--final_json_root",
        type=str,
        help="post-Step-3 sidecar JSON directory for the same batch",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        help="Output path used with sidecar JSON mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.initial_json_root and args.final_json_root and args.output_json):
        raise SystemExit(
            "--initial_json_root, --final_json_root, and --output_json are required"
        )

    result = ShapeMismatchRepairEvaluator().summarize_root_cause_pair(
        initial_json_root=args.initial_json_root,
        final_json_root=args.final_json_root,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
