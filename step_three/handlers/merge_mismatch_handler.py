from typing import Any
from loguru import logger
from pprint import pformat
from step_three.handlers.base import LLMRepairHandler
from step_three.llm_runtime import LLMRuntime
from step_three.merge_mismatch_repair.helper.code_rewriter import build_repaired_code
from step_three.merge_mismatch_repair.params_solver import MismatchParametersSolver
from step_three.prompts.merge_mismatch_prompt import MergeMismatchPromptDesigner
from step_three.result import RepairResult



class MergeMismatchHandler(LLMRepairHandler):
    """Implement the merge mismatch handler component."""

    SUPPORTED_ROOT_CAUSE_TYPES = {"Tensor_Merge_Mismatch"}
    PROMPT_DESIGNER_CLASS = MergeMismatchPromptDesigner

    def __init__(
        self,
        step2_result: dict[str, Any],
        llm_runtime: LLMRuntime,
        merge_solver: MismatchParametersSolver,
    ):
        super().__init__(
            step2_result=step2_result,
            llm_runtime=llm_runtime,
        )
        self.merge_solver = merge_solver

    def solve_candidate_space(
        self,
        llm_response: Any,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Solve candidate space."""

        candidate_space = self.merge_solver.candidate_adapter.build_candidate_space(
            llm_response
        )
        solver_result = self.merge_solver.main(
            step2_result=self.step2_result,
            candidate_space=candidate_space,
        )

        return candidate_space, solver_result

    def build_repair_meta(
        self,
        candidate_space: list[dict[str, Any]],
        solver_result: list[dict[str, Any]],
        code_replacements: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build repair meta."""
        root_cause = self.step2_result.get("root_cause") or {}
        if solver_result:
            solver_status = "solved"
        elif not candidate_space:
            solver_status = "no_candidate"
        else:
            solver_status = "no_valid_candidate"

        return {
            "masked_api_calls": root_cause.get("target_layers") or [],
            "candidate_space": candidate_space,
            "solver_status": solver_status,
            "solver_result": solver_result,
            "code_replacements": code_replacements or [],
        }

    def rewrite_repaired_code(
        self,
        solver_result: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Rewrite repaired code."""
        root_cause = self.step2_result.get("root_cause") or {}
        source_code = self.step2_result.get("final_code", "")
        buggy_api_calls = root_cause.get("target_layers") or []

        if not solver_result:
            return self.step2_result.get("origin_code", ""), []

        return build_repaired_code(
            source_code=source_code,
            buggy_api_calls=buggy_api_calls,
            solver_result=solver_result,
        )

    def main(self) -> RepairResult:
        """Run the main workflow."""
        root_cause_type = self.get_root_cause_type()
        logger.info(f"Root-cause type: {root_cause_type}")

        prompt, response = self.request_llm_with_prompt()
        logger.info(f"Prompt:\n{prompt}")
        logger.info(f"LLM response:\n{response}")

        candidate_space, solver_result = self.solve_candidate_space(response)
        logger.info(f"Candidate parameter space:\n{candidate_space}")
        solver_result_text = pformat(solver_result, width=200, sort_dicts=False)
        logger.info(f"Parameter-solving result:\n{solver_result_text}")

        repaired_code, code_replacements = self.rewrite_repaired_code(solver_result)
        logger.info(f"Fully repaired code:\n{repaired_code}")

        return RepairResult(
            root_cause_type=root_cause_type,
            prompt=prompt,
            llm_response=response,
            repaired_code=repaired_code,
            repair_meta=self.build_repair_meta(
                candidate_space,
                solver_result,
                code_replacements=code_replacements,
            ),
        )
