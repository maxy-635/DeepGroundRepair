import argparse
import json
import os
import re
import sys
import subprocess
from loguru import logger
from pathlib import Path
import pandas as pd
from utils.df2excel import DataFrame2Excel
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Validation:
    """Implement the validation component."""

    @staticmethod
    def sort_key(text):
        """Sort key."""
        return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(text))]

    @staticmethod
    def is_valid_pyfile(pyfile):
        with open(pyfile, "r", encoding="utf-8") as file:
            return any(line.strip() for line in file)

    def compile_code(self, pyfile):

        logger.info(f"Checking {pyfile}...")

        result = subprocess.run([sys.executable, "-W", "ignore", str(pyfile)], capture_output=True, text=True)

        if result.returncode == 0:
            compile_status = "compile success"
            compile_error = None
        else:
            compile_status = "compile failed"
            compile_error = result.stderr or result.stdout

        verification = {
            "pyfile_path": str(pyfile),
            "compile_status": str(compile_status),
            "compile_error": str(compile_error),
        }

        return verification

    def evaluate_step(self, code_path):
        """Evaluate step."""
        verifications, rows = [], []
        task_dirs = sorted([p for p in code_path.iterdir() if p.is_dir()], key=self.sort_key)

        for task_dir in task_dirs:
            valid_pyfiles = [
                p for p in sorted(task_dir.glob("*.py"), key=self.sort_key)
                if self.is_valid_pyfile(p)
            ]
            task_results = [self.compile_code(p) for p in valid_pyfiles]
            verifications.extend(task_results)

            success = sum(item["compile_status"] == "compile success" for item in task_results)
            total = len(valid_pyfiles)
            rows.append({
                "Benchmark": task_dir.name,
                "success": success,
                "failed": total - success,
                "valid_pyfiles": total,
                "rate(%)": 100 * success / total if total else 0,
            })

        df_data = pd.DataFrame(rows)
        total = {
            "Benchmark": "Total",
            "success": df_data["success"].sum() if not df_data.empty else 0,
            "failed": df_data["failed"].sum() if not df_data.empty else 0,
            "valid_pyfiles": df_data["valid_pyfiles"].sum() if not df_data.empty else 0,
            "rate(%)": 0,
        }
        if total["valid_pyfiles"]:
            total["rate(%)"] = 100 * total["success"] / total["valid_pyfiles"]

        return verifications, pd.concat([df_data, pd.DataFrame([total])], ignore_index=True)

    def save_step_report(self, verifications, df_data, report_dir, excel_path, step):
        """Save step report."""
        report_dir.mkdir(parents=True, exist_ok=True)
        with open(report_dir / f"{step}.json", "w", encoding="utf-8") as file:
            json.dump(verifications, file, ensure_ascii=False, indent=2)
        DataFrame2Excel(df_data, str(excel_path)).df2excel(sheet_name=step)

    def run(self, response_root, report_root, llms, steps, dlls, experiment_batch):
        """Run the operation."""
        for llm in llms:
            for dll in dlls:
                dll = dll.lower()
                summary_rows = []
                report_dir = report_root / llm / dll
                excel_path = report_dir / "dynamic_checking.xlsx"

                for step in steps:
                    code_path = response_root / llm / step / dll
                    logger.info(f"Dynamic Checking: llm={llm}, step={step}, dll={dll}, batch={experiment_batch}")

                    verifications, df_data = self.evaluate_step(code_path)
                    self.save_step_report(verifications, df_data, report_dir, excel_path, step)

                    total_row = df_data.iloc[-1].copy()
                    total_row["Benchmark"] = step
                    summary_rows.append(total_row)

                if summary_rows:
                    summary_df = pd.DataFrame(summary_rows).reset_index(drop=True)
                    logger.info(summary_df.to_string(index=False))
                    DataFrame2Excel(summary_df, str(excel_path)).df2excel(sheet_name="summary")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=str, required=True, help="JSON list of step ids")
    parser.add_argument("--llms", type=str, required=True, help="JSON list of model ids")
    parser.add_argument("--dlls", type=str, required=True, help="JSON list of dll names")
    parser.add_argument("--experiment_id", type=str, required=True, help="experiment batch id")
    args = parser.parse_args()

    llms = json.loads(args.llms)
    steps = json.loads(args.steps)
    dlls = json.loads(args.dlls)
    experiment_batch = args.experiment_id

    Validation().run(
        response_root=Path("response"),
        report_root=Path("evaluation/dynamic_checking/report"),
        llms=llms,
        steps=steps,
        dlls=dlls,
        experiment_batch=experiment_batch,
    )
