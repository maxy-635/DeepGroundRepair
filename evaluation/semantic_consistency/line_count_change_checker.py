import json
from collections import Counter
from pathlib import Path
from typing import Any
from utils.count_code_lines import count_code_lines



class LineCountChangeChecker:
    """Compare source/repaired files by effective code-line count."""

    def evaluate_dirs(self, source_dir: Path | str, repaired_dir: Path | str) -> dict[str, Any]:
        source_dir = Path(source_dir)
        repaired_dir = Path(repaired_dir)

        source_files = self._task_file_map(source_dir)
        repaired_files = self._task_file_map(repaired_dir)
        paired_tasks = sorted(set(source_files) & set(repaired_files))

        samples = []
        excluded_empty_source_cases = []
        for task_id in paired_tasks:
            source_path = source_files[task_id]
            repaired_path = repaired_files[task_id]
            source_lines = count_code_lines(source_path)
            repaired_lines = count_code_lines(repaired_path)
            if source_lines == 0:
                excluded_empty_source_cases.append(
                    {
                        "task_id": task_id,
                        "source_path": str(source_path),
                        "repaired_path": str(repaired_path),
                        "source_code_lines": source_lines,
                        "repaired_code_lines": repaired_lines,
                        "repaired_code_is_empty": repaired_lines == 0,
                    }
                )
                continue

            delta = repaired_lines - source_lines

            samples.append(
                {
                    "task_id": task_id,
                    "source_path": str(source_path),
                    "repaired_path": str(repaired_path),
                    "source_code_lines": source_lines,
                    "repaired_code_lines": repaired_lines,
                    "line_delta": delta,
                    "change_type": self._change_type(delta),
                    "has_line_addition": delta > 0,
                    "has_line_deletion": delta < 0,
                }
            )

        change_counts = Counter(sample["change_type"] for sample in samples)
        semantic_consistency_rate = (
            change_counts.get("unchanged", 0) / len(samples)
            if samples
            else 0
        )
        empty_source_repaired_empty_cases = sum(
            case["repaired_code_is_empty"] for case in excluded_empty_source_cases
        )
        return {
            "source_dir": str(source_dir),
            "repaired_dir": str(repaired_dir),
            "summary": {
                "paired_tasks": len(paired_tasks),
                "valid_files": len(samples),
                "excluded_empty_source_cases": len(excluded_empty_source_cases),
                "all_empty_source_repaired_empty": (
                    len(excluded_empty_source_cases) == empty_source_repaired_empty_cases
                ),
                "change_counts": dict(sorted(change_counts.items())),
                "line_added_cases": change_counts.get("added", 0),
                "line_deleted_cases": change_counts.get("deleted", 0),
                "line_unchanged_cases": change_counts.get("unchanged", 0),
                "semantic_consistency_rate": semantic_consistency_rate,
            },
            "samples": samples,
            "excluded_empty_source_cases": excluded_empty_source_cases,
        }

    def _task_file_map(self, root: Path) -> dict[str, Path]:
        task_files = {}
        for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            py_files = sorted(task_dir.glob("*.py"))
            if py_files:
                task_files[task_dir.name] = py_files[0]
        return task_files

    def _change_type(self, delta: int) -> str:
        if delta > 0:
            return "added"
        if delta < 0:
            return "deleted"
        return "unchanged"





if __name__ == "__main__":
    llms = [
        "gemma_3_4b_it",
        "gemma_3_12b_it",
        "gemma_3_27b_it",
        "ministral_3_3b_instruct_2512",
        "ministral_3_14b_instruct_2512",
        "mistral_small_3_2_24b_instruct_2506",
    ]

    comparison_pairs = [
        ("step_one", "step_three"),
    ]
    dlls = ["TensorFlow", "PyTorch", "PaddlePaddle"]
    experiment_batch = None  # compatibility placeholder; response trees are flattened
    response_root = Path("response")
    report_dir = Path("evaluation/semantic_consistency/report")

    dll_dir_names = {
        "TensorFlow": "tensorflow",
        "PyTorch": "pytorch",
        "PaddlePaddle": "paddlepaddle",
    }
    checker = LineCountChangeChecker()

    all_summaries = []
    for source_step, repaired_step in comparison_pairs:
        for llm in llms:
            for dll in dlls:
                dll_dir = dll_dir_names[dll]
                source_dir = response_root / llm / source_step / dll_dir
                repaired_dir = response_root / llm / repaired_step / dll_dir
                if not source_dir.exists() or not repaired_dir.exists():
                    print(f"Skip missing path: {source_step}->{repaired_step}, {llm}, {dll_dir}")
                    continue

                result = checker.evaluate_dirs(source_dir, repaired_dir)
                pair_dir = f"{source_step}_2_{repaired_step}"
                output_path = report_dir / pair_dir / llm / f"{dll_dir}.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                summary = {
                    "source_step": source_step,
                    "repaired_step": repaired_step,
                    "llm": llm,
                    "dll": dll_dir,
                    **result["summary"],
                }
                all_summaries.append(summary)
                print(json.dumps(summary, ensure_ascii=False, indent=2))

    summary_path = report_dir / "line_count_summary.json"
    summary_path.write_text(
        json.dumps(all_summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
