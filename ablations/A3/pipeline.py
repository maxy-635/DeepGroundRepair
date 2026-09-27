from pathlib import Path
from loguru import logger
from ablations import common
from ablations.A3.context import build_no_shape_context
from ablations.A3.prompting import NoShapeContextPromptDesigner
from step_three.llm_runtime import LLMRuntime
from step_three.result import RepairResult
from utils.utils import get_all_files


class NoShapeContextRepairPipeline:
    """Implement the no shape context repair pipeline component."""

    def __init__(
        self,
        step2_result_path: str | Path,
        repair_code_save_path: str | Path,
        llm_runtime: LLMRuntime,
    ) -> None:
        self.step2_result_path = Path(step2_result_path)
        self.repair_code_save_path = Path(repair_code_save_path)
        self.llm_runtime = llm_runtime

    def get_root_cause_type(self, step2_result: dict) -> str:
        """Return root cause type."""
        root_cause = step2_result.get("root_cause") or {}
        root_cause_type = root_cause.get("root_cause_type")
        return str(root_cause_type)

    def build_prompt(self, step2_result: dict) -> str:
        """Build prompt."""
        prompt_context = build_no_shape_context(step2_result)
        prompt_designer = NoShapeContextPromptDesigner(prompt_context)
        return prompt_designer.prompt()

    def request_llm_with_prompt(self, step2_result: dict) -> tuple[str, object]:
        """Request llm with prompt."""
        prompt = self.build_prompt(step2_result)
        response = self.llm_runtime.chat(prompt)
        return prompt, response

    def extract_repaired_code(self, step2_result: dict, response: object) -> str:
        """Extract repaired code."""
        repaired_code = common.extract_python_code(response)
        if repaired_code:
            return repaired_code
        return build_no_shape_context(step2_result)["masked_code"]

    def repair_one(self, step2_result: dict) -> RepairResult:
        root_cause_type = self.get_root_cause_type(step2_result)
        prompt, response = self.request_llm_with_prompt(step2_result)

        logger.info(f"Prompt:\n{prompt}")
        logger.info(f"LLM response:\n{response}")

        repaired_code = self.extract_repaired_code(step2_result, response)

        return RepairResult(
            root_cause_type=root_cause_type,
            prompt=prompt,
            llm_response=response,
            repaired_code=repaired_code,
        )

    def build_repair_code_path(self, step2_json_file: str | Path) -> Path:
        """Build repair code path."""
        relative = common.safe_relative_to(step2_json_file, self.step2_result_path)
        if relative.name.endswith("_annotated.json"):
            repaired_name = relative.name.removesuffix("_annotated.json") + "_repaired.py"
            relative = relative.with_name(repaired_name)
        else:
            relative = relative.with_suffix(".py")
        return self.repair_code_save_path / relative

    def save_repaired_code(self, repair_code_path: Path, repaired_code: str) -> None:
        """Save repaired code."""
        repair_code_path.parent.mkdir(parents=True, exist_ok=True)
        repair_code_path.write_text(repaired_code, encoding="utf-8")

    def main(self) -> None:
        """Run the main workflow."""
        for json_file in get_all_files(str(self.step2_result_path), ".json"):
            logger.info(f"[A3] Reading Step Two result: {json_file}")
            step2_result = common.load_json(json_file)
            repair_code_path = self.build_repair_code_path(json_file)

            root_cause = step2_result.get("root_cause") or {}
            root_cause_type = root_cause.get("root_cause_type")
            if root_cause_type == "No_Runtime_Error":
                self.save_repaired_code(repair_code_path, step2_result.get("final_code", ""))
                logger.info(f"[A3] Saving original code for the case without runtime errors: {repair_code_path}")
                continue
            
            repair_result = self.repair_one(step2_result)
            # Saving repaired code
            self.save_repaired_code(repair_code_path, repair_result.repaired_code)

            logger.info(f"[A3] Saving repaired code: {repair_code_path}")
