import ast
import copy
import importlib
import inspect


class RewriteNotApplicable(Exception):
    """Implement the rewrite not applicable component."""

    def __init__(self, reason: str, resolved_callable: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.resolved_callable = resolved_callable


def collect_import_references(source_code: str):
    """Collect import references."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return {}

    import_references = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "*":
                    continue

                if alias.asname:
                    import_references[alias.asname] = alias.name
                else:
                    root_name = alias.name.split(".")[0]
                    import_references[root_name] = root_name

        if isinstance(node, ast.ImportFrom):
            if node.module is None or node.level != 0:
                continue

            for alias in node.names:
                if alias.name == "*":
                    continue

                local_name = alias.asname or alias.name
                import_references[local_name] = f"{node.module}.{alias.name}"

    return import_references


def resolve_runtime_callable(func_node: ast.AST, import_references, resolved_modules=None):
    """Resolve runtime callable."""
    dotted_parts = []
    current = func_node
    while isinstance(current, ast.Attribute):
        dotted_parts.append(current.attr)
        current = current.value

    if not isinstance(current, ast.Name):
        return None, None

    dotted_parts.append(current.id)
    dotted_parts.reverse()

    root_name = dotted_parts[0]
    if root_name in import_references:
        resolved_parts = import_references[root_name].split(".") + dotted_parts[1:]
    else:
        resolved_parts = dotted_parts

    resolved_name = ".".join(resolved_parts)

    if resolved_modules is None:
        resolved_modules = {}

    module_name = resolved_parts[0]
    if module_name not in resolved_modules:
        try:
            resolved_modules[module_name] = importlib.import_module(module_name)
        except Exception:
            resolved_modules[module_name] = None

    obj = resolved_modules[module_name]
    if obj is None:
        return None, resolved_name

    try:
        for attr in resolved_parts[1:]:
            obj = getattr(obj, attr)
    except AttributeError:
        return None, resolved_name

    if not callable(obj):
        return None, resolved_name

    return obj, resolved_name


def extract_target_call(stmt: ast.stmt):
    """Extract target call."""
    if isinstance(stmt, ast.Assign):
        value = stmt.value
    elif isinstance(stmt, ast.AnnAssign):
        value = stmt.value
    elif isinstance(stmt, ast.Expr):
        value = stmt.value
    elif isinstance(stmt, ast.Return):
        value = stmt.value
    else:
        return None

    if not isinstance(value, ast.Call):
        return None

    if isinstance(value.func, ast.Call):
        return value.func

    return value


def has_unsupported_unpacking(call_node: ast.Call):
    if any(isinstance(arg, ast.Starred) for arg in call_node.args):
        return True

    return any(keyword.arg is None for keyword in call_node.keywords)


def map_positional_arguments_to_parameters(call_node: ast.Call, signature: inspect.Signature):
    """Map positional arguments to parameters."""
    positional_markers = [("__positional__", index) for index, _ in enumerate(call_node.args)]
    keyword_markers = {
        keyword.arg: ("__keyword__", keyword.arg)
        for keyword in call_node.keywords
    }

    try:
        bound = signature.bind_partial(*positional_markers, **keyword_markers)
    except TypeError as exc:
        raise RewriteNotApplicable("bind_partial_failed") from exc

    marker_to_param = {}
    for param_name, value in bound.arguments.items():
        parameter = signature.parameters[param_name]

        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            if value:
                raise RewriteNotApplicable("var_positional_parameter")
            continue

        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            continue

        if isinstance(value, tuple) and len(value) == 2 and value[0] == "__positional__":
            if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
                raise RewriteNotApplicable("positional_only_parameter")
            marker_to_param[value[1]] = param_name

    if len(marker_to_param) != len(positional_markers):
        raise RewriteNotApplicable("positional_mapping_incomplete")

    return [marker_to_param[index] for index in range(len(positional_markers))]


def validate_keyword_arguments(call_node: ast.Call, signature: inspect.Signature):
    """Validate keyword arguments."""
    for keyword in call_node.keywords:
        if keyword.arg is None:
            raise RewriteNotApplicable("contains_unpacking")

        parameter = signature.parameters.get(keyword.arg)
        if parameter is None:
            continue

        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            raise RewriteNotApplicable("keyword_on_positional_only_parameter")


def build_rewritten_keywords(
    call_node: ast.Call,
    positional_param_names,
    mask_values: bool,
    mask_value: str,
    mask_parameter_names=None,
):
    """Build rewritten keywords."""
    rewritten_keywords = []

    for param_name, arg_node in zip(positional_param_names, call_node.args):
        should_mask = (
            mask_values
            and (mask_parameter_names is None or param_name in mask_parameter_names)
        )
        if should_mask:
            rewritten_value = ast.Constant(value=mask_value)
        else:
            rewritten_value = copy.deepcopy(arg_node)
        rewritten_keywords.append(ast.keyword(arg=param_name, value=rewritten_value))

    for keyword in call_node.keywords:
        should_mask = (
            mask_values
            and (mask_parameter_names is None or keyword.arg in mask_parameter_names)
        )
        if should_mask:
            rewritten_value = ast.Constant(value=mask_value)
        else:
            rewritten_value = copy.deepcopy(keyword.value)
        rewritten_keywords.append(ast.keyword(arg=keyword.arg, value=rewritten_value))

    return rewritten_keywords
