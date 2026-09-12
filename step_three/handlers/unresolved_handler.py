from typing import Any
from loguru import logger
from step_three.handlers.base import LLMRepairHandler
from step_three.prompts.unresolved_crash_prompt import UnresolvedCrashPromptDesigner
from step_three.result import RepairResult


class UnresolvedCrashHandler(LLMRepairHandler):
    """Implement the unresolved crash handler component."""

    SUPPORTED_ROOT_CAUSE_TYPES = {"Unresolved_Crash"}
    PROMPT_DESIGNER_CLASS = UnresolvedCrashPromptDesigner

    def extract_repaired_code(self, llm_response: Any) -> str:
        """Extract repaired code."""
        if llm_response is None:
            return ""

        response_text = str(llm_response).strip()
        return self.extract_first_code_block(response_text, language="python")

    def main(self) -> RepairResult:
        """Run the main workflow."""
        root_cause_type = self.get_root_cause_type()
        logger.info(f"Root-cause type: {root_cause_type}")

        prompt, response = self.request_llm_with_prompt()
        logger.info(f"Prompt:\n{prompt}")
        logger.info(f"LLM response:\n{response}")

        repaired_code = self.extract_repaired_code(response)
        logger.info(f"Fully repaired code:\n{repaired_code}")
        
        return RepairResult(
            root_cause_type=root_cause_type,
            prompt=prompt,
            llm_response=response,
            repaired_code=repaired_code,
        )