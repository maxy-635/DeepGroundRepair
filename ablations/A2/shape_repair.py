from pathlib import Path
from loguru import logger
from ablations import common
from ablations.A2.shape_annotation import (
    NO_RUNTIME_ERROR_TYPE,
    ROOT_CAUSE_FREE_TYPE,
    SHAPE_ANNOTATION_UNAVAILABLE_TYPE,
)
from ablations.A2.shape_repair_prompt import ShapeOnlyRepairPromptDesigner
from step_three.llm_runtime import LLMRuntime
from utils.utils import get_all_files


SUPPORTED_REPAIR_ROOT_CAUSE_TYPES = {
    ROOT_CAUSE_FREE_TYPE,
    SHAPE_ANNOTATION_UNAVAILABLE_TYPE,
}


class ShapeOnlyRepairPipeline:
    """Implement the shape only repair pipeline component."""

    def __init__(
        self,
        step2_result_path: str | Path,
        repair_code_save_path: str | Path,
        llm_runtime: LLMRuntime,
    ) -> None:
        self.step2_result_path = Path(step2_result_path)
        self.repair_code_save_path = Path(repair_code_save_path)
        self.llm_runtime = llm_runtime

    @staticmethod
    def save_repaired_code(path: Path, code: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")

    def build_repair_code_path(self, step2_json_file: str | Path) -> Path:
        relative = common.safe_relative_to(step2_json_file, self.step2_result_path)
        repaired_name = relative.name.replace("_annotated.json", "_repaired.py")
        if repaired_name == relative.name:
            repaired_name = relative.with_suffix(".py").name
        return self.repair_code_save_path / relative.parent / repaired_name

    def main(self) -> None:
        """Run the main workflow."""

        for json_file in get_all_files(str(self.step2_result_path), ".json"):
            logger.info(f"[A2/Step3] Reading Step Two result: {json_file}")
            step2_result = common.load_json(json_file)
            repair_code_path = self.build_repair_code_path(json_file)

            root_cause_type = (step2_result.get("root_cause") or {}).get("root_cause_type")
            if root_cause_type == NO_RUNTIME_ERROR_TYPE:
                self.save_repaired_code(repair_code_path, step2_result.get("final_code", ""))
                logger.info(f"[A2/Step3] Saving the case without runtime errors directly: {repair_code_path}")
                continue

            if root_cause_type not in SUPPORTED_REPAIR_ROOT_CAUSE_TYPES:
                logger.info(f"[A2/Step3] Skipping a case type unsupported by A2 ({root_cause_type}): {json_file}")
                continue

            prompt = ShapeOnlyRepairPromptDesigner(step2_result).prompt()
            response = self.llm_runtime.chat(prompt)
            logger.info(f"Prompt:\n{prompt}")
            logger.info(f"LLM response:\n{response}")

            repaired_code = common.extract_python_code(response)
            if not repaired_code:
                logger.error(f"[A2/Step3] No Python code was extracted from the LLM response; skipping: {json_file}")
                continue

            self.save_repaired_code(repair_code_path, repaired_code)
            logger.info(f"[A2/Step3] Saving repaired code: {repair_code_path}")
