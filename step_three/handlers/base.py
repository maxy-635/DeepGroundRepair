import re
from typing import Any
from step_three.llm_runtime import LLMRuntime


class LLMRepairHandler:
    """Implement the llmrepair handler component."""

    SUPPORTED_ROOT_CAUSE_TYPES: set[str] = set()
    PROMPT_DESIGNER_CLASS: type | None = None

    def __init__(self, step2_result: dict[str, Any], llm_runtime: LLMRuntime) -> None:
        self.step2_result = step2_result
        self.llm_runtime = llm_runtime
        self.model_id = self.llm_runtime.model_id

    def get_root_cause_type(self) -> str:
        """Return root cause type."""
        root_cause = self.step2_result.get("root_cause") or {}
        root_cause_type = root_cause.get("root_cause_type")
        if root_cause_type not in self.SUPPORTED_ROOT_CAUSE_TYPES:
            supported = ", ".join(sorted(self.SUPPORTED_ROOT_CAUSE_TYPES))
            raise ValueError(
                f"{type(self).__name__} supports [{supported}], got: {root_cause_type}"
            )
        return str(root_cause_type)

    def build_prompt(self) -> str:
        """Build prompt."""
        if self.PROMPT_DESIGNER_CLASS is None:
            raise NotImplementedError("PROMPT_DESIGNER_CLASS must be configured")
        return self.PROMPT_DESIGNER_CLASS(self.step2_result).prompt()

    def request_llm_with_prompt(self) -> tuple[str, Any]:
        """Request llm with prompt."""
        prompt = self.build_prompt()
        response = self.llm_runtime.chat(prompt)

        return prompt, response

    @staticmethod
    def extract_first_code_block(llm_response: Any, language: str) -> str | None:
        """Extract first code block."""

        response_text = str(llm_response).strip()
        if language:
            pattern = rf"```{re.escape(language)}\s*(.*?)```"
        else:
            pattern = r"```(?:\w+)?\s*(.*?)```"

        matched_code = re.search(pattern, response_text, re.S)
        if matched_code:
            return matched_code.group(1).strip()