import re
from typing import Any
from loguru import logger
from step_three.handlers.base import LLMRepairHandler
from step_three.merge_mismatch_repair.helper.code_rewriter import split_inline_comment
from step_three.prompts.direct_repair_prompt import DirectRepairPromptDesigner
from step_three.result import RepairResult


class DirectRepairHandler(LLMRepairHandler):
    """Implement the direct repair handler component."""

    SUPPORTED_ROOT_CAUSE_TYPES = {"General_Crash", "Forward_Local_Crash"}
    PROMPT_DESIGNER_CLASS = DirectRepairPromptDesigner

    def extract_code_block(self, llm_response: Any) -> str | None:
        """Extract code block."""
        repaired_api_call = self.extract_first_code_block(llm_response, language="python")
        if repaired_api_call:
            return repaired_api_call

        response_text = str(llm_response).replace("<end_of_turn>", "").strip()
        repaired_api_call = None

        for line in response_text.splitlines():
            candidate = line.strip()
            if not candidate:
                continue
            candidate = candidate.strip("`").strip()
            if candidate.startswith("- "):
                candidate = candidate[2:].strip()
            if self.get_call_head(candidate):
                repaired_api_call = candidate
                break

        return repaired_api_call

    def strip_inline_comment(self, code_line: str) -> str:
        """Strip inline comment."""
        code_part, _ = split_inline_comment(code_line)
        return code_part.strip()

    def get_call_head(self, code_line: str) -> str | None:
        """Return call head."""
        code_part = self.strip_inline_comment(code_line).rstrip(",").strip()
        equal_index = code_part.find("=")
        paren_index = code_part.find("(")
        if equal_index != -1 and (paren_index == -1 or equal_index < paren_index):
            code_part = code_part.split("=", 1)[1].strip()

        match = re.match(r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(", code_part)
        return match.group(1) if match else None

    def normalize_replacement_line(
        self,
        repaired_api_call: str,
        buggy_api_call: dict[str, Any],
        original_line: str,
    ) -> str:
        """Normalize replacement line."""
        replacement_line = repaired_api_call.strip()
        masked_call_head = self.get_call_head(
            buggy_api_call.get("masked_api_call", "")
        )

        if "\n" in replacement_line and masked_call_head:
            for candidate_line in replacement_line.splitlines():
                candidate = candidate_line.strip()
                if self.get_call_head(candidate) == masked_call_head:
                    replacement_line = candidate
                    break

        original_code = self.strip_inline_comment(original_line)
        original_has_comma = original_code.endswith(",")
        replacement_has_comma = self.strip_inline_comment(replacement_line).endswith(",")
        if original_has_comma and not replacement_has_comma:
            replacement_line = f"{replacement_line},"

        return replacement_line

    def extract_buggy_api_call(self) -> dict[str, Any]:
        """Extract buggy api call."""
        buggy_api_calls = DirectRepairPromptDesigner(
            self.step2_result
        ).extract_buggy_api_calls()
        return buggy_api_calls[0] if buggy_api_calls else {}

    def find_rewrite_line_index(
        self,
        lines: list[str],
        buggy_api_call: dict[str, Any],
    ) -> int | None:
        """Find rewrite line index."""

        masked_code_part = self.strip_inline_comment(
            buggy_api_call.get("masked_api_call", "")
        )
        if masked_code_part:
            for index, line in enumerate(lines):
                if self.strip_inline_comment(line) == masked_code_part:
                    return index

        line_no = buggy_api_call.get("line_no")
        if isinstance(line_no, int) and 1 <= line_no <= len(lines):
            return line_no - 1

        return None

    def rewrite_repaired_code(
        self,
        repaired_api_call: str | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Rewrite repaired code."""
        source_code = self.step2_result.get("final_code", "")
        replacement_line = repaired_api_call.strip() if repaired_api_call else ""
        if not replacement_line:
            return source_code, []

        lines = source_code.splitlines()
        buggy_api_call = self.extract_buggy_api_call()
        line_index = self.find_rewrite_line_index(
            lines=lines,
            buggy_api_call=buggy_api_call,
        )
        if line_index is None:
            return source_code, []

        original_line = lines[line_index]
        replacement_line = self.normalize_replacement_line(
            repaired_api_call=replacement_line,
            buggy_api_call=buggy_api_call,
            original_line=original_line,
        )
        indentation = original_line[: len(original_line) - len(original_line.lstrip())]
        lines[line_index] = f"{indentation}{replacement_line}"

        replacement = {
            "buggy_api_call_index": buggy_api_call.get("buggy_api_call_index", 0),
            "line_no": line_index + 1,
            "original_line": original_line.strip(),
            "replacement": replacement_line,
        }
        return "\n".join(lines), [replacement]

    def main(self) -> RepairResult:
        """Run the main workflow."""
        root_cause_type = self.get_root_cause_type()
        logger.info(f"Root-cause type: {root_cause_type}")

        prompt, response = self.request_llm_with_prompt()
        logger.info(f"Prompt:\n{prompt}")
        logger.info(f"LLM response:\n{response}")

        repaired_api_call = self.extract_code_block(response)
        logger.info(f"Repaired buggy API call: {repaired_api_call}")

        repaired_code, code_replacements = self.rewrite_repaired_code(
            repaired_api_call=repaired_api_call
        )
        logger.info(f"Fully repaired code:\n{repaired_code}")

        return RepairResult(
            root_cause_type=root_cause_type,
            prompt=prompt,
            llm_response=response,
            repaired_code=repaired_code,
            repair_meta={
                "repaired_api_call": repaired_api_call,
                "code_replacements": code_replacements,
            },
        )
