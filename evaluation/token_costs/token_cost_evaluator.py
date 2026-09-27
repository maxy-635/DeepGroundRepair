import json
import re
import argparse
from pathlib import Path


class LLMTokenLogParser:
    """Parse per-case LLM token counts from experiment logs."""

    RESPONSE_PATTERN = re.compile(r"LLM generated response tokens:\s*(\d+)")
    CASE_PATTERNS = [
        re.compile(r"(?:Start processing task|\u5f00\u59cb\u5904\u7406\u4efb\u52a1):\s*((?:easy|medium|hard)_task_\d+)"),
        re.compile(r"Processing (?:pyfile|file):\s*.*[/\\]((?:easy|medium|hard)_task_\d+)[/\\]"),
    ]

    def __init__(self, log_file, dll=None):
        self.log_file = Path(log_file)
        self.dll = dll

    def parse(self):
        cases = {}
        current_case = None

        with self.log_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                case_match = None
                for pattern in self.CASE_PATTERNS:
                    case_match = pattern.search(line)
                    if case_match:
                        break
                if case_match:
                    current_case = case_match.group(1)
                    cases.setdefault(current_case, self.case_summary(current_case))
                    continue

                token_match = self.RESPONSE_PATTERN.search(line)
                if token_match and current_case:
                    case = cases.setdefault(current_case, self.case_summary(current_case))
                    case["llm_calls"] += 1
                    case["response_tokens"] += int(token_match.group(1))

        case_list = list(cases.values())
        log_summary = {
            "log_file": str(self.log_file),
            "dll": self.dll,
            "cases": len(case_list),
            "llm_calls": sum(item["llm_calls"] for item in case_list),
            "response_tokens": sum(item["response_tokens"] for item in case_list),
        }
        return log_summary, case_list

    def case_summary(self, case_name):
        return {
            "case": case_name,
            "log_file": str(self.log_file),
            "dll": self.dll,
            "llm_calls": 0,
            "response_tokens": 0,
        }


class LLMTokenLogCounter:
    """Aggregate LLM token counts across experiment logs."""

    def __init__(self, log_dir, output_json):
        self.log_dir = Path(log_dir)
        self.output_json = Path(output_json)

    def run(self):
        summary = self.summarize()
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        with self.output_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary

    def summarize(self):
        if not self.log_dir.exists():
            raise FileNotFoundError(f"Log directory does not exist: {self.log_dir}")
        if not self.log_dir.is_dir():
            raise NotADirectoryError(f"Log path is not a directory: {self.log_dir}")

        per_log = []
        per_case = []
        log_files = sorted(self.log_dir.rglob("*.log"), key=lambda path: [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))])
        if not log_files:
            raise FileNotFoundError(f"No .log files found in the log directory: {self.log_dir}")

        for log_file in log_files:
            dll = self.infer_dll(log_file)
            log_summary, case_summaries = LLMTokenLogParser(log_file, dll=dll).parse()
            per_log.append(log_summary)
            per_case.extend(case_summaries)

        return {
            "log_file": str(self.log_dir),
            "per_log": per_log,
            "case_average": {
                "all_cases": self.average(per_case),
                "cases_with_llm": self.average([item for item in per_case if item["llm_calls"] > 0]),
                "by_dll": self.average_by_dll(per_case),
            },
            "total": {
                "cases": len(per_case),
                "llm_calls": sum(item["llm_calls"] for item in per_log),
                "response_tokens": sum(item["response_tokens"] for item in per_log),
            },
        }

    def infer_dll(self, log_file):
        relative_parts = log_file.relative_to(self.log_dir).parts
        if len(relative_parts) > 1:
            return relative_parts[0].lower()
        return "unknown"

    def average_by_dll(self, cases):
        dll_cases = {}
        for case in cases:
            dll_cases.setdefault(case["dll"], []).append(case)

        return {
            dll: {
                "all_cases": self.average(group_cases),
                "cases_with_llm": self.average([item for item in group_cases if item["llm_calls"] > 0]),
            }
            for dll, group_cases in sorted(dll_cases.items())
        }

    def average(self, cases):
        if not cases:
            return {"cases": 0, "avg_llm_calls": 0, "avg_response_tokens": 0}
        return {
            "cases": len(cases),
            "avg_llm_calls": sum(item["llm_calls"] for item in cases) / len(cases),
            "avg_response_tokens": sum(item["response_tokens"] for item in cases) / len(cases),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--LLMs", type=str, required=True)
    parser.add_argument("--step_id", type=str, required=True)
    parser.add_argument("--experiment_id", type=str, required=True)
    args = parser.parse_args()

    llms = json.loads(args.LLMs)
    for llm in llms:
        log_dir = f"logs/{llm}/{args.step_id}"
        output_json = f"evaluation/token_costs/report/{args.step_id}/{llm}.json"

        LLMTokenLogCounter(log_dir=log_dir, output_json=output_json).run()
