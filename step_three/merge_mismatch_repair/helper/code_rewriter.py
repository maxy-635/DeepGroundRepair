from __future__ import annotations

import ast
import io
import tokenize
from typing import Any


def build_literal_node(value: Any) -> ast.expr:
    """Build literal node."""
    return ast.parse(repr(value), mode="eval").body


def find_nested_call_node(node: ast.AST) -> ast.Call | None:
    """Find nested call node."""
    if isinstance(node, ast.Assign):
        return find_nested_call_node(node.value)
    if isinstance(node, ast.Expr):
        return find_nested_call_node(node.value)
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Call):
            return find_nested_call_node(node.func)
        return node
    return None


def instantiate_masked_call(masked_api_call: str, solved_params: dict[str, Any]) -> str:
    """Instantiate masked call."""
    tree = ast.parse(masked_api_call)
    if not tree.body:
        raise ValueError("empty masked_api_call")

    call_node = find_nested_call_node(tree.body[0])
    if call_node is None:
        raise ValueError(f"masked_api_call is not a supported call: {masked_api_call}")

    updated_params = set()
    for keyword in call_node.keywords:
        if keyword.arg in solved_params:
            keyword.value = build_literal_node(solved_params[keyword.arg])
            updated_params.add(keyword.arg)

    missing_params = sorted(set(solved_params) - updated_params)
    if missing_params:
        raise ValueError(
            f"solved params not found in masked_api_call: {', '.join(missing_params)}"
        )

    ast.fix_missing_locations(tree)
    return ast.unparse(tree.body[0])


def replace_code_line(
    source_code: str,
    line_no: int,
    replacement: str,
) -> str:
    """Replace code line."""
    lines = source_code.splitlines()
    if not 1 <= line_no <= len(lines):
        raise ValueError(f"line_no out of range: {line_no}")

    original_line = lines[line_no - 1]
    indentation = original_line[: len(original_line) - len(original_line.lstrip())]
    suffix = extract_replacement_suffix(original_line)
    lines[line_no - 1] = f"{indentation}{replacement}{suffix}"
    return "\n".join(lines)


def extract_replacement_suffix(original_line: str) -> str:
    """Extract replacement suffix."""
    stripped_line = original_line.lstrip()
    code_part, comment_part = split_inline_comment(stripped_line)
    suffix = ""

    code_without_trailing_space = code_part.rstrip()
    trailing_space = code_part[len(code_without_trailing_space) :]
    if code_without_trailing_space.endswith(","):
        suffix += ","

    if comment_part:
        suffix += trailing_space + comment_part

    return suffix


def split_inline_comment(line: str) -> tuple[str, str]:
    """Split inline comment."""
    try:
        tokens = tokenize.generate_tokens(io.StringIO(line).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                comment_start = token.start[1]
                return line[:comment_start], line[comment_start:]
    except tokenize.TokenError:
        pass

    return line, ""


def build_repaired_code(
    source_code: str,
    buggy_api_calls: list[dict[str, Any]],
    solver_result: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Build repaired code."""
    buggy_api_calls_by_index = {
        index: buggy_api_call
        for index, buggy_api_call in enumerate(buggy_api_calls)
        if isinstance(buggy_api_call, dict)
    }
    repaired_code = source_code
    replacements = []

    for solved_layer in sorted(
        solver_result,
        key=lambda item: item.get("buggy_api_call_index", -1),
    ):
        buggy_api_call_index = solved_layer.get("buggy_api_call_index")
        if not isinstance(buggy_api_call_index, int):
            raise ValueError(f"invalid buggy_api_call_index: {buggy_api_call_index}")

        buggy_api_call = buggy_api_calls_by_index.get(buggy_api_call_index)
        if buggy_api_call is None:
            raise ValueError(
                f"missing buggy API call for index: {buggy_api_call_index}"
            )

        line_no = buggy_api_call.get("line_no")
        if not isinstance(line_no, int):
            raise ValueError(f"invalid buggy API call line_no: {line_no}")

        solved_params = solved_layer.get("solved_params")
        if not isinstance(solved_params, dict):
            raise ValueError(
                f"invalid solved_params for buggy API call {buggy_api_call_index}"
            )

        masked_api_call = buggy_api_call.get("masked_api_call")
        if not isinstance(masked_api_call, str) or not masked_api_call.strip():
            raise ValueError(
                f"missing masked_api_call for buggy API call {buggy_api_call_index}"
            )

        replacement = instantiate_masked_call(masked_api_call, solved_params)
        repaired_code = replace_code_line(
            source_code=repaired_code,
            line_no=line_no,
            replacement=replacement,
        )
        replacements.append(
            {
                "buggy_api_call_index": buggy_api_call_index,
                "line_no": line_no,
                "solved_params": solved_params,
                "replacement": replacement,
            }
        )

    return repaired_code, replacements
