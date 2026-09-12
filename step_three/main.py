import argparse
import json
from pathlib import Path
from typing import Any
from loguru import logger
from step_three.handlers import (
    DirectRepairHandler,
    MergeMismatchHandler,
    UnresolvedCrashHandler,
)
from step_three.llm_runtime import LLMRuntime
from step_three.merge_mismatch_repair.params_solver import MismatchParametersSolver
from step_three.result import RepairResult
from utils.utils import get_all_files, get_llm_name



class CodeRegenerationPipeline:
    """Implement the code regeneration pipeline component."""

    def __init__(
        self, step2_result_path: str, repair_code_save_path: str, llm_runtime: LLMRuntime
    ) -> None:
        self.step2_result_path = Path(step2_result_path)
        self.repair_code_save_path = Path(repair_code_save_path) if repair_code_save_path else None
        self.llm_runtime = llm_runtime
        self.merge_solver = MismatchParametersSolver()

    def validate_step2_result(self, step2_result: dict[str, Any]) -> None:
        """Validate step2 result."""
        required_keys = ("final_code", "runtime_error_info", "root_cause")
        for key in required_keys:
            if key not in step2_result:
                raise ValueError(f"Missing required Step 2 field: {key}")

        root_cause = step2_result.get("root_cause") or {}
        if "root_cause_type" not in root_cause:
            raise ValueError("Missing required field: root_cause.root_cause_type")
        if (
            root_cause.get("root_cause_type") == "Tensor_Merge_Mismatch"
            and not step2_result.get("origin_code")
        ):
            raise ValueError(
                "Missing required Step 2 field for Tensor_Merge_Mismatch: origin_code"
            )

    def build_repair_code_path(self, step2_result_file: str) -> Path:
        """Build repair code path."""
        step2_file = Path(step2_result_file)

        relative_file = step2_file.relative_to(self.step2_result_path)
        repair_filename = f"{step2_file.stem.replace('_annotated', '')}_repaired.py"

        return self.repair_code_save_path / relative_file.with_name(repair_filename)

    def repair(self, step2_result: dict[str, Any]) -> RepairResult:
        self.validate_step2_result(step2_result)

        root_cause_type = step2_result.get("root_cause").get("root_cause_type")

        if root_cause_type == "No_Runtime_Error":
            logger.info("Root-cause type: No_Runtime_Error")
            return RepairResult(
                root_cause_type=root_cause_type,
                repaired_code=step2_result["final_code"],
            )

        if root_cause_type in {"General_Crash", "Forward_Local_Crash"}:
            direct_repair_handler = DirectRepairHandler(
                step2_result=step2_result,
                llm_runtime=self.llm_runtime,
            )
            return direct_repair_handler.main()

        if root_cause_type == "Tensor_Merge_Mismatch":
            merge_mismatch_handler = MergeMismatchHandler(
                step2_result=step2_result,
                merge_solver=self.merge_solver,
                llm_runtime=self.llm_runtime,
            )
            return merge_mismatch_handler.main()

        if root_cause_type == "Unresolved_Crash":
            unresolved_crash_handler = UnresolvedCrashHandler(
                step2_result=step2_result,
                llm_runtime=self.llm_runtime,
            )
            return unresolved_crash_handler.main()

        raise ValueError(f"Unsupported root cause type: {root_cause_type}")



    def main(self) -> dict[str, Any]:
        """Run the main workflow."""
        step2_result_files = get_all_files(self.step2_result_path, ".json")
        logger.info(f"Found {len(step2_result_files)} Step 2 result files in {self.step2_result_path}")

        for step2_result_file in step2_result_files:
            logger.info(f"Processing file: {step2_result_file}")

            with open(step2_result_file, "r", encoding="utf-8") as file:
                step2_result = json.load(file)

            repair_result = self.repair(step2_result=step2_result)
            save_repair_pycode_file = self.build_repair_code_path(step2_result_file)

            save_repair_pycode_file.parent.mkdir(parents=True, exist_ok=True)
            with open(save_repair_pycode_file, "w", encoding="utf-8") as pyfile:
                if repair_result.repaired_code:
                    pyfile.write(repair_result.repaired_code)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--step2_result_path", type=str, required=True)
    parser.add_argument("--repair_code_save_path", type=str, required=True)
    parser.add_argument("--experiment_id", type=str, required=True)
    parser.add_argument("--dlls", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dlls_list = json.loads(args.dlls) if args.dlls else []
    llm_name = get_llm_name(args.model_id)

    sampling_config_sink_ids = []
    for dll in dlls_list:
        dll_key = dll.lower()
        log_file = f"logs/{llm_name}/step_three/{dll_key}/{args.experiment_id}.log"
        sampling_config_sink_ids.append(
            logger.add(
                log_file,
                filter=lambda record: record["extra"].get("sampling_config", False),
            )
        )

    llm_runtime = LLMRuntime.hf(model_id=args.model_id, cache_dir=args.cache_dir)

    for sink_id in sampling_config_sink_ids:
        logger.remove(sink_id)

    step2_root = Path(args.step2_result_path)
    step3_root = Path(args.repair_code_save_path)

    log_sink_id = None
    for dll in dlls_list:
        dll_key = dll.lower()
        step2_result_path = step2_root / dll.lower()
        repair_code_save_path = step3_root / dll.lower()

        if log_sink_id is not None:
            logger.remove(log_sink_id)

        log_file = f"logs/{llm_name}/step_three/{dll_key}/{args.experiment_id}.log"
        log_sink_id = logger.add(log_file)

        if not step2_result_path.exists():
            logger.warning(f"Step Two result path does not exist; skipping DLL {dll}: {step2_result_path}")
            continue

        logger.info(
            f"Processing DLL: {dll}, Model ID: {llm_name}, Experiment ID: {args.experiment_id}, "
            f"Step Two path: {step2_result_path}, Repaired-code output path: {repair_code_save_path}"
        )

        step_three_pipeline = CodeRegenerationPipeline(
            step2_result_path=str(step2_result_path),
            repair_code_save_path=str(repair_code_save_path),
            llm_runtime=llm_runtime,
        )
        step_three_pipeline.main()
