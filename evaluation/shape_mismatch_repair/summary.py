from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


METHOD_DIRS = {
    "full": "full",
    "v3": "v3",
}

DLLS = ("tensorflow", "pytorch", "paddlepaddle")

INITIAL_BUCKETS = ("Intra-path", "Inter-path", "Others")
FINAL_BUCKETS = ("Fixed", "Intra-path", "Inter-path", "Others")


def initial_bucket(root_cause_type: str | None) -> str:
    if root_cause_type == "General_Crash":
        return "Intra-path"
    if root_cause_type == "Tensor_Merge_Mismatch":
        return "Inter-path"
    return "Others"


def final_bucket(root_cause_type: str | None) -> str:
    if root_cause_type == "No_Runtime_Error":
        return "Fixed"
    if root_cause_type == "General_Crash":
        return "Intra-path"
    if root_cause_type == "Tensor_Merge_Mismatch":
        return "Inter-path"
    return "Others"


def load_per_sample_records(report_root: Path, method: str, dll: str) -> list[dict]:
    """Load and merge per-sample records across the two models for one DLL."""
    records: list[dict] = []
    method_root = report_root / METHOD_DIRS[method]
    for json_path in sorted(method_root.glob(f"*/{dll}.json")):
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        records.extend(payload.get("per_sample", []))
    return records


def build_transition_matrix(records: Iterable[dict]) -> pd.DataFrame:
    counts = defaultdict(Counter)
    for sample in records:
        row = initial_bucket(sample.get("initial_root_cause_type"))
        col = final_bucket(sample.get("final_root_cause_type"))
        counts[row][col] += 1
        counts["All shape error"][col] += 1

    index = [*INITIAL_BUCKETS, "All shape error"]
    columns = [*FINAL_BUCKETS, "All"]

    data = []
    for row_name in index:
        row_counts = counts[row_name]
        row = [row_counts.get(col, 0) for col in FINAL_BUCKETS]
        row.append(sum(row))
        data.append(row)

    df = pd.DataFrame(data, index=index, columns=columns)
    df.index.name = "Initial \\ Final"
    return df


def add_sheet(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame, note: str) -> None:
    df.to_excel(writer, sheet_name=sheet_name)
    ws = writer.book[sheet_name]
    ws.insert_rows(1)
    ws["A1"] = note
    ws.freeze_panes = "B4"
    for column_cells in ws.columns:
        max_len = 0
        col_letter = column_cells[0].column_letter
        for cell in column_cells:
            value = "" if cell.value is None else str(cell.value)
            max_len = max(max_len, len(value))
        ws.column_dimensions[col_letter].width = min(max_len + 2, 28)


def summarize_and_write(report_root: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    note = (
        "Initial: General_Crash -> Intra-path, Tensor_Merge_Mismatch -> Inter-path, "
        "others -> Others; Final: No_Runtime_Error -> Fixed."
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for method in METHOD_DIRS:
            for dll in DLLS:
                records = load_per_sample_records(report_root, method, dll)
                matrix = build_transition_matrix(records)
                sheet_name = {
                    ("full", "tensorflow"): "full_tf",
                    ("full", "pytorch"): "full_torch",
                    ("full", "paddlepaddle"): "full_paddle",
                    ("v3", "tensorflow"): "v3_tf",
                    ("v3", "pytorch"): "v3_torch",
                    ("v3", "paddlepaddle"): "v3_paddle",
                }[(method, dll)]
                add_sheet(writer, sheet_name, matrix, note)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report_root",
        type=Path,
        default=Path("evaluation/shape_mismatch_repair/report/RQ3"),
        help="Root directory containing full/ and v3/ RQ3 reports",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/shape_mismatch_repair/report/RQ3/root_cause_transition_matrices.xlsx"),
        help="Output Excel file path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = summarize_and_write(args.report_root, args.output)
    print(out)


if __name__ == "__main__":
    main()
