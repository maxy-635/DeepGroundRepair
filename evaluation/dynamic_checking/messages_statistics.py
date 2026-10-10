import re
import json
import pandas as pd
from collections import Counter
from loguru import logger
from pathlib import Path
from evaluation.dynamic_checking.my_compile_utils.effectiveness import ComputeValidCode


class RootCauseDataReader:
    """Implement the root cause data reader component."""

    def __init__(self, model_id, dlls, step, experiment_id, detail_steps=None, response_root="response"):
        self.model_id = model_id
        self.dlls = dlls
        self.step = step
        self.experiment_id = experiment_id
        self.detail_steps = detail_steps or [step]
        self.response_root = Path(response_root)
        self.step_summaries = {}
        self.detail_root_cause_types = {}
        self.task_names = {}
        self._root_cause_cache = {}

    @staticmethod
    def sort_key(text):
        return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(text))]

    def response_dir(self, dll, step):
        return self.response_root / self.model_id / step / dll

    def root_cause_type(self, json_file):
        if json_file not in self._root_cause_cache:
            with open(json_file, "r", encoding="utf-8") as file:
                data = json.load(file)
            self._root_cause_cache[json_file] = (
                (data.get("root_cause") or {}).get("root_cause_type", "Unknown")
            )
        return self._root_cause_cache[json_file]

    def read(self):
        step_summaries = {
            step: {"dll_counters": {}, "total_counter": Counter(), "valid_pyfiles": {}}
            for step in self.detail_steps
        }
        detail_root_cause_types = {}
        task_names_by_dll = {}

        for dll in self.dlls:
            task_names = set()
            for step in self.detail_steps:
                step_dir = self.response_dir(dll, step)
                if step_dir.exists():
                    task_names.update(path.name for path in step_dir.iterdir() if path.is_dir())

            task_names_by_dll[dll] = sorted(task_names, key=self.sort_key)
            detail_root_cause_types[dll] = {}
            for step in self.detail_steps:
                step_dir = self.response_dir(dll, step)
                if not step_dir.exists():
                    logger.info(f"Path does not exist; skipping: {step_dir}")
                    counter = Counter()
                    valid_pyfiles = 0
                else:
                    json_files = sorted(
                        step_dir.rglob("*.json"),
                        key=lambda path: self.sort_key(str(path.relative_to(step_dir))),
                    )
                    counter = Counter()
                    for json_file in json_files:
                        counter.update([self.root_cause_type(json_file)])
                    valid_pyfiles = ComputeValidCode().valid_code(str(step_dir))

                step_summaries[step]["dll_counters"][dll] = counter
                step_summaries[step]["total_counter"].update(counter)
                step_summaries[step]["valid_pyfiles"][dll] = valid_pyfiles

                detail_root_cause_types[dll][step] = {}
                for task_name in task_names_by_dll[dll]:
                    json_files = sorted(
                        (step_dir / task_name).glob("*.json"),
                        key=lambda path: self.sort_key(path.name),
                    )
                    detail_root_cause_types[dll][step][task_name] = "; ".join(
                        self.root_cause_type(json_file)
                        for json_file in json_files
                    )

        for summary in step_summaries.values():
            summary["valid_pyfiles"]["Total"] = sum(summary["valid_pyfiles"].values())

        self.step_summaries = step_summaries
        self.detail_root_cause_types = detail_root_cause_types
        self.task_names = task_names_by_dll


class MarkdownRootCauseWriter:
    """Implement the markdown root cause writer component."""

    def __init__(self, model_id, step, experiment_id, report_root="evaluation/dynamic_checking/report"):
        self.model_id = model_id
        self.step = step
        self.experiment_id = experiment_id
        self.report_root = Path(report_root)

    def build_summary_df(self, read_result):
        summary = read_result.step_summaries[self.step]
        root_cause_types = sorted(summary["total_counter"].keys())
        rows = []

        for dll, counter in {**summary["dll_counters"], "Total": summary["total_counter"]}.items():
            row = {root_cause_type: counter.get(root_cause_type, 0) for root_cause_type in root_cause_types}
            row["Runtime_Error"] = sum(
                count for root_cause_type, count in counter.items()
                if root_cause_type != "No_Runtime_Error"
            )
            row["valid_pyfiles"] = summary["valid_pyfiles"].get(dll, 0)
            rows.append({"dll": dll, **row})

        df = pd.DataFrame(rows)
        required_columns = ["dll", "No_Runtime_Error", "Runtime_Error", "valid_pyfiles"]
        df = df.reindex(columns=df.columns.union(required_columns), fill_value=0)

        first_columns = ["dll", "No_Runtime_Error", "Runtime_Error"]
        last_columns = ["valid_pyfiles"]
        middle_columns = [column for column in df.columns if column not in first_columns + last_columns]

        return df[first_columns + middle_columns + last_columns]

    @staticmethod
    def dataframe_to_text(df):
        rows = [df.columns.tolist()] + df.astype(str).values.tolist()
        col_widths = [max(len(str(row[index])) for row in rows) for index in range(len(df.columns))]

        def format_row(values):
            return "| " + " | ".join(str(value).rjust(col_widths[index]) for index, value in enumerate(values)) + " |"

        line = "+-" + "-+-".join("-" * width for width in col_widths) + "-+"
        return "\n".join([line, format_row(rows[0]), line, *[format_row(row) for row in rows[1:]], line])

    @staticmethod
    def dataframe_to_markdown(df):
        headers = ["DLL" if column == "dll" else column for column in df.columns]
        lines = [
            "| " + " | ".join(f"**{header}**" for header in headers) + " |",
            "|---|" + "|".join("---:" for _ in headers[1:]) + "|",
        ]

        for row in df.astype(str).values.tolist():
            if row[0] == "Total":
                row = [f"**{cell}**" for cell in row]
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def save(self, read_result):
        save_dir = self.report_root / self.model_id
        save_dir.mkdir(parents=True, exist_ok=True)

        summary_df = self.build_summary_df(read_result)
        table_text = self.dataframe_to_text(summary_df)
        save_path = save_dir / f"{self.step}.md"
        save_path.write_text(self.dataframe_to_markdown(summary_df), encoding="utf-8")

        return table_text, save_path


class ExcelRootCauseWriter:
    """Implement the excel root cause writer component."""

    def __init__(self, model_id, dlls, step, experiment_id, report_root="evaluation/dynamic_checking/report"):
        self.model_id = model_id
        self.dlls = dlls
        self.step = step
        self.experiment_id = experiment_id
        self.report_root = Path(report_root)

    def save(self, read_result):
        save_paths = []

        for dll in self.dlls:
            save_dir = self.report_root / self.model_id / dll
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "root_cause_types.xlsx"

            benchmarks = read_result.task_names.get(dll, [])
            root_cause_by_task = read_result.detail_root_cause_types.get(dll, {}).get(self.step, {})
            df = pd.DataFrame(
                [{"Benchmark": benchmark, "root_cause_type": root_cause_by_task.get(benchmark, "")}
                 for benchmark in benchmarks],
                columns=["Benchmark", "root_cause_type"],
            )

            with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name=self.step, index=False)
                sheet = writer.sheets[self.step]
                sheet.freeze_panes = "A2"
                sheet.column_dimensions["A"].width = 24
                sheet.column_dimensions["B"].width = 34

            save_paths.append(save_path)

        return save_paths


def run(model_id, dlls, step, experiment_id, detail_steps=None):
    reader = RootCauseDataReader(model_id, dlls, step, experiment_id, detail_steps)
    reader.read()

    table_text = ""
    md_save_paths = []
    xlsx_save_paths = []
    for step_name in reader.detail_steps:
        table_text, md_save_path = MarkdownRootCauseWriter(model_id, step_name, experiment_id).save(reader)
        md_save_paths.append(md_save_path)
        xlsx_save_paths.extend(ExcelRootCauseWriter(model_id, dlls, step_name, experiment_id).save(reader))

    return table_text, md_save_paths, xlsx_save_paths


if __name__ == "__main__":
    model_id = "gemma_3_27b_it"
    step = "step_two"
    detail_steps = ["step_two", "step_three"]
    dlls = ["TensorFlow", "PyTorch", "PaddlePaddle"]
    experiment_id = "****"

    table_text, md_save_paths, xlsx_save_paths = run(model_id, dlls, step, experiment_id, detail_steps)

    logger.info(f"\n{model_id} root_cause_type, Runtime_Error, and valid Python file counts:")
    logger.info(table_text)
    logger.info("\nStatistics saved to:")
    for md_save_path in md_save_paths:
        logger.info(md_save_path)
    logger.info("\nroot_cause_type details saved to:")
    for xlsx_save_path in xlsx_save_paths:
        logger.info(xlsx_save_path)
