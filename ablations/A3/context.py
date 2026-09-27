import re
from typing import Any


SHAPE_COMMENT_PATTERNS = (
    re.compile(r"\s*#\s*the initial shape of .*$"),
    re.compile(r"\s*#\s*output tensor shape.*$"),
)


def build_no_shape_context(step2_result: dict[str, Any]) -> dict[str, Any]:
    """Build no shape context."""
    return {
        "masked_code": strip_shape_context_from_code(step2_result.get("final_code", "")),
    }


def strip_shape_context_from_code(code: str) -> str:
    """Strip shape context from code."""
    cleaned_lines = [
        strip_shape_context_from_line(line).rstrip()
        for line in code.splitlines()
    ]
    cleaned_code = "\n".join(cleaned_lines).strip()
    return f"{cleaned_code}\n" if cleaned_code else ""


def strip_shape_context_from_line(line: str) -> str:
    cleaned = line
    for pattern in SHAPE_COMMENT_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned
