import io
import tokenize
import ast
from dataclasses import dataclass, field

from step_two.root_cause_masking.api_call_rewriter import ApiCallRewriter
from step_two.root_cause_masking.general_crash_linear_masking import (
    mask_linear_call_statically,
    select_general_crash_params_to_mask,
)


@dataclass
class InitModuleDefinition:
    """Implement the init module definition component."""

    module_name: str
    line_no: int
    code_line: str
    module_type: str
    end_line_no: int | None = None
    children: list = field(default_factory=list)


@dataclass
class ModuleUseDefMapping:
    """Implement the module use def mapping component."""

    crash_location: object
    called_module_name: str
    init_definition: InitModuleDefinition


def split_inline_comment(code_line: str) -> tuple[str, str]:
    """Split inline comment."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code_line).readline))
    except tokenize.TokenError:
        return code_line.rstrip(), ""

    for token in tokens:
        if token.type == tokenize.COMMENT:
            code_part = code_line[: token.start[1]].rstrip()
            inline_comment = code_line[token.start[1] :].strip()
            return code_part, inline_comment
    return code_line.rstrip(), ""


def mask_existing_init_argument_values(code_line: str) -> str:
    """Mask existing init argument values."""
    try:
        tree = ast.parse(code_line)
    except SyntaxError:
        return code_line.strip()

    if not tree.body:
        return code_line.strip()

    stmt = tree.body[0]
    call_node = getattr(stmt, "value", None)
    if not isinstance(call_node, ast.Call):
        return code_line.strip()

    call_node.args = [
        ast.copy_location(ast.Constant(value="mask"), arg)
        for arg in call_node.args
    ]
    call_node.keywords = [
        ast.keyword(
            arg=keyword.arg,
            value=ast.copy_location(ast.Constant(value="mask"), keyword.value),
        )
        for keyword in call_node.keywords
    ]
    ast.fix_missing_locations(tree)
    return ast.unparse(stmt)


def build_masked_init_api_call(
    init_definition,
    source_code: str | None = None,
    params_to_mask: list[str] | None = None,
) -> str:
    """Build masked init api call."""
    original_call = init_definition.code_line
    code_without_comment, inline_comment = split_inline_comment(original_call)

    masked_call = None
    if source_code:
        rewrite_result = ApiCallRewriter().rewrite(
            source_code=source_code,
            code_line=code_without_comment,
            mask_parameter_names=params_to_mask or None,
        )
        if rewrite_result.failure_reason is None:
            masked_call = rewrite_result.masked_call

    if masked_call is None:
        if params_to_mask:
            masked_call = mask_linear_call_statically(
                code_line=code_without_comment,
                params_to_mask=params_to_mask,
            )
        if masked_call is None:
            masked_call = mask_existing_init_argument_values(code_without_comment)

    if inline_comment:
        return f"{masked_call} {inline_comment}"
    return masked_call


def build_general_root_cause_result(mapping, source_code: str | None = None):
    """Build general root cause result."""
    crash_location = mapping.crash_location
    init_definition = mapping.init_definition
    params_to_mask = select_general_crash_params_to_mask(
        mapping=mapping,
        source_code=source_code,
    )

    init_definition_result = {
        "line_no": init_definition.line_no,
        "end_line_no": init_definition.end_line_no,
        "original_call": init_definition.code_line,
        "masked_api_call": build_masked_init_api_call(
            init_definition=init_definition,
            source_code=source_code,
            params_to_mask=params_to_mask,
        ),
    }
    if params_to_mask:
        init_definition_result["params_to_mask"] = params_to_mask

    return {
        "crash_location": {
            "crash_stage": getattr(crash_location, "crash_stage", "unknown"),
            "line_no": getattr(crash_location, "line_no", None),
            "crash_code": getattr(crash_location, "crash_code", ""),
        },
        "init_definition": init_definition_result,
        "root_cause_type": "General_Crash",
    }


def build_unresolved_root_cause_result(crash_location):
    """Build unresolved root cause result."""
    return {
        "crash_location": {
            "crash_stage": getattr(crash_location, "crash_stage", "unknown"),
            "line_no": getattr(crash_location, "line_no", None),
            "crash_code": getattr(crash_location, "crash_code", ""),
        },
        "init_definition": {},
        "root_cause_type": "Unresolved_Crash",
    }


def get_root_cause_original_call(root_cause):
    """Return root cause original call."""
    if root_cause.get("root_cause_type") == "Tensor_Merge_Mismatch":
        target_layers = root_cause.get("target_layers") or []
        primary_target_layer_index = root_cause.get("primary_target_layer_index")
        if isinstance(primary_target_layer_index, int):
            if 0 <= primary_target_layer_index < len(target_layers):
                primary_target_layer = target_layers[primary_target_layer_index]
                if primary_target_layer.get("original_call"):
                    return primary_target_layer["original_call"]

    init_definition = root_cause.get("init_definition") or {}
    if init_definition.get("original_call"):
        return init_definition["original_call"]

    forward_definition = root_cause.get("forward_definition") or {}
    if forward_definition.get("original_call"):
        return forward_definition["original_call"]

    crash_location = root_cause.get("crash_location") or {}
    return crash_location.get("crash_code", "")
