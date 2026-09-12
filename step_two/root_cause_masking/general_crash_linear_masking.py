import ast


LINEAR_INPUT_FEATURE_PARAM_ALIASES = {"in_features", "in_channels"}


def detect_source_framework(source_code: str | None) -> str | None:
    """Best-effort framework detection from source imports."""
    if not source_code:
        return None

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        lowered_source = source_code.lower()
        if "paddle" in lowered_source:
            return "paddle"
        if "torch" in lowered_source:
            return "pytorch"
        return None

    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])

    if "paddle" in imported_roots:
        return "paddle"
    if "torch" in imported_roots:
        return "pytorch"
    return None


def select_general_crash_params_to_mask(mapping, source_code: str | None = None) -> list[str]:
    """Only mask Linear's input-feature parameter for forward General_Crash."""
    crash_location = mapping.crash_location
    init_definition = mapping.init_definition

    if getattr(crash_location, "crash_stage", "unknown") != "forward":
        return []
    if getattr(init_definition, "module_type", "") != "Linear":
        return []

    framework = detect_source_framework(source_code)
    if framework == "paddle":
        # Some Paddle versions/docs use in_features, while project notes may call
        # the same slot in_channels. Keep both so the installed signature decides
        # which keyword actually appears in the rewritten call.
        return ["in_channels", "in_features"]
    if framework == "pytorch":
        return ["in_features"]
    return ["in_features"]


def mask_linear_call_statically(code_line: str, params_to_mask: list[str]) -> str | None:
    """Fallback selective Linear masking when runtime signature resolution fails."""
    try:
        tree = ast.parse(code_line)
    except SyntaxError:
        return None

    if not tree.body:
        return None

    stmt = tree.body[0]
    value = getattr(stmt, "value", None)
    if not isinstance(value, ast.Call):
        return None

    func = value.func
    api_name = None
    if isinstance(func, ast.Attribute):
        api_name = func.attr
    elif isinstance(func, ast.Name):
        api_name = func.id
    if api_name != "Linear":
        return None

    params_to_mask_set = set(params_to_mask)
    preferred_input_param = (
        "in_channels"
        if "in_channels" in params_to_mask_set
        else "in_features"
    )

    rewritten_keywords = []
    positional_names = [preferred_input_param, "out_features"]
    for index, arg_node in enumerate(value.args):
        if index >= len(positional_names):
            return None
        param_name = positional_names[index]
        if _should_mask_param(param_name, params_to_mask_set):
            rewritten_value = ast.Constant(value="mask")
        else:
            rewritten_value = arg_node
        rewritten_keywords.append(ast.keyword(arg=param_name, value=rewritten_value))

    for keyword in value.keywords:
        if keyword.arg is None:
            return None
        if _should_mask_param(keyword.arg, params_to_mask_set):
            keyword.value = ast.Constant(value="mask")
        rewritten_keywords.append(keyword)

    value.args = []
    value.keywords = rewritten_keywords
    ast.fix_missing_locations(tree)
    return ast.unparse(stmt)


def _should_mask_param(param_name: str, params_to_mask_set: set[str]) -> bool:
    if param_name in params_to_mask_set:
        return True
    return (
        param_name in LINEAR_INPUT_FEATURE_PARAM_ALIASES
        and bool(params_to_mask_set & LINEAR_INPUT_FEATURE_PARAM_ALIASES)
    )
