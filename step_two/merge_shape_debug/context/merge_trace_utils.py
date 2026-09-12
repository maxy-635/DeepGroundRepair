import ast
import re
from step_two.merge_shape_debug.context.merge_ast_utils import parse_merge_crash_code
from step_two.tensor_shape_trace.tensor_shape_tracer import TensorShapeTrace


FORWARD_LIKE_FUNCTION_NAMES = ("forward", "call")
PREFERRED_FORWARD_INPUT_NAMES = ("x", "input", "inputs")


def build_visitor(source_code, source_path):
    """Build visitor."""
    tracer = TensorShapeTrace(python_code=source_code, filename=source_path)
    return tracer.build_ast()


def build_merge_context(
    visitor,
    traced_shapes,
    initial_input_shapes,
    crash_location,
    init_definitions,
    runtime_error_info,
    source_code=None,
):
    """Build merge context."""

    merge_info = parse_merge_crash_code(crash_location.crash_code, init_definitions, source_code=source_code)
    if merge_info is None:
        return None
    forward_input_names, direct_input_aliases, latest_assignments = collect_forward_input_context(
        source_code=source_code,
        crash_location=crash_location,
    )
    forward_input_shape_map = resolve_forward_input_shape_map(
        source_code=source_code,
        crash_location=crash_location,
        initial_input_shapes=initial_input_shapes or {},
    )

    branches = []
    statement_kind = merge_info.get("statement_kind", "assign")
    accumulator_var_name = None
    if statement_kind == "augassign" and merge_info["input_vars"]:
        accumulator_var_name = merge_info["input_vars"][0]

    for index, var_name in enumerate(merge_info["input_vars"]):
        producer_path = extract_producer_path(visitor, var_name, crash_location.line_no)
        producer_calls = extract_producer_calls(
            visitor,
            producer_path,
            init_definitions,
            traced_shapes=traced_shapes,
            forward_input_shape_map=forward_input_shape_map,
        )
        connected_forward_inputs = resolve_connected_forward_inputs(
            var_name=var_name,
            producer_path=producer_path,
            forward_input_names=forward_input_names,
        )
        input_connection_type = infer_input_connection_type(
            var_name=var_name,
            connected_forward_inputs=connected_forward_inputs,
            direct_input_aliases=direct_input_aliases,
            latest_assignments=latest_assignments,
            producer_calls=producer_calls,
        )
        branch_shape = resolve_shape_before_merge(
            traced_shapes=traced_shapes,
            var_name=var_name,
            merge_lineno=crash_location.line_no,
        )
        if branch_shape is None:
            branch_shape = resolve_shape_from_initial_inputs(
                var_name=var_name,
                connected_forward_inputs=connected_forward_inputs,
                input_connection_type=input_connection_type,
                forward_input_shape_map=forward_input_shape_map,
            )
        branches.append(
            {
                "var_name": var_name,
                "shape": branch_shape,
                "producer_path": producer_path,
                "producer_calls": producer_calls,
                "path_length": len(producer_calls),
                "role": "unknown",
                "connected_forward_inputs": connected_forward_inputs,
                "input_connection_type": input_connection_type,
                "is_direct_input_alias": input_connection_type == "direct_alias",
                "is_accumulator_branch": statement_kind == "augassign" and index == 0,
            }
        )

    fill_missing_shapes_from_error_message(branches, runtime_error_info)
    annotate_branch_roles(branches, forward_input_names=forward_input_names)

    return {
        "merge_lineno": crash_location.line_no,
        "merge_code": crash_location.crash_code,
        "merge_kind": merge_info["merge_kind"],
        "concat_axis": merge_info["concat_axis"],
        "statement_kind": statement_kind,
        "accumulator_var_name": accumulator_var_name,
        "forward_input_names": list(forward_input_names),
        "forward_input_shapes": forward_input_shape_map,
        "branches": branches,
    }


def collect_forward_input_context(source_code, crash_location):
    """Collect forward input context."""
    if not source_code:
        return [], set(), {}

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return [], set(), {}

    function_node = find_relevant_forward_function_node(tree, crash_location)
    if function_node is None:
        return [], set(), {}

    forward_input_names = extract_forward_input_names(function_node)
    direct_input_aliases = collect_direct_input_aliases(
        function_node=function_node,
        seed_input_names=forward_input_names,
        stop_lineno=getattr(crash_location, "line_no", None),
    )
    latest_assignments = collect_latest_assignments(
        function_node=function_node,
        stop_lineno=getattr(crash_location, "line_no", None),
    )
    return forward_input_names, direct_input_aliases, latest_assignments


def resolve_forward_input_shape_map(source_code, crash_location, initial_input_shapes):
    """Resolve forward input shape map."""
    if not source_code or not initial_input_shapes:
        return {}

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return {}

    function_node = find_relevant_forward_function_node(tree, crash_location)
    if function_node is None:
        return {}

    shape_items = initial_input_shapes.get(function_node.lineno, [])
    result = {}
    for item in shape_items:
        var_name = item.get("var_name")
        shape = item.get("shape")
        if var_name is None or shape is None:
            continue
        result[var_name] = shape
    return result


def find_relevant_forward_function_node(tree, crash_location):
    """Find relevant forward function node."""
    function_nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not function_nodes:
        return None

    candidate_names = []
    crash_func_name = getattr(crash_location, "func_name", None)
    if crash_func_name in FORWARD_LIKE_FUNCTION_NAMES:
        candidate_names.append(crash_func_name)
    for name in FORWARD_LIKE_FUNCTION_NAMES:
        if name not in candidate_names:
            candidate_names.append(name)

    line_no = getattr(crash_location, "line_no", None)
    for function_name in candidate_names:
        for node in function_nodes:
            if node.name != function_name:
                continue
            if is_lineno_inside_node(line_no, node):
                return node

    for function_name in candidate_names:
        for node in function_nodes:
            if node.name == function_name:
                return node

    return None


def is_lineno_inside_node(line_no, node):
    if line_no is None:
        return False
    end_lineno = getattr(node, "end_lineno", node.lineno)
    return node.lineno <= line_no <= end_lineno


def extract_forward_input_names(function_node):
    """Extract forward input names."""
    ordered = []
    seen = set()

    def append_name(name):
        if not name or name == "self" or name in seen:
            return
        seen.add(name)
        ordered.append(name)

    positional_args = list(getattr(function_node.args, "posonlyargs", [])) + list(function_node.args.args)
    for arg in positional_args:
        append_name(arg.arg)

    if function_node.args.vararg is not None:
        append_name(function_node.args.vararg.arg)

    for arg in function_node.args.kwonlyargs:
        append_name(arg.arg)

    if function_node.args.kwarg is not None:
        append_name(function_node.args.kwarg.arg)

    preferred = [name for name in PREFERRED_FORWARD_INPUT_NAMES if name in seen]
    others = [name for name in ordered if name not in preferred]
    return preferred + others


def collect_direct_input_aliases(function_node, seed_input_names, stop_lineno=None):
    """Collect direct input aliases."""
    aliases = set(seed_input_names)
    pure_alias_assignments = extract_pure_alias_assignments(function_node, stop_lineno=stop_lineno)

    changed = True
    while changed:
        changed = False
        for target_name, source_name in pure_alias_assignments:
            if source_name not in aliases or target_name in aliases:
                continue
            aliases.add(target_name)
            changed = True

    return aliases


def extract_pure_alias_assignments(function_node, stop_lineno=None):
    """Extract pure alias assignments."""
    assignments = []
    for node in iter_relevant_function_nodes(function_node):
        if isinstance(node, ast.Assign):
            if stop_lineno is not None and node.lineno >= stop_lineno:
                continue
            source_name = extract_plain_name(node.value)
            if source_name is None:
                continue
            for target in node.targets:
                for target_name in extract_plain_target_names(target):
                    assignments.append((target_name, source_name))
            continue

        if isinstance(node, ast.AnnAssign):
            if stop_lineno is not None and node.lineno >= stop_lineno:
                continue
            source_name = extract_plain_name(node.value)
            if source_name is None:
                continue
            target_name = extract_plain_name(node.target)
            if target_name is not None:
                assignments.append((target_name, source_name))

    return assignments


def collect_latest_assignments(function_node, stop_lineno=None):
    """Collect latest assignments."""
    assignments = []
    for node in iter_relevant_function_nodes(function_node):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        node_lineno = getattr(node, "lineno", None)
        if stop_lineno is not None and node_lineno is not None and node_lineno >= stop_lineno:
            continue
        assignments.append(node)

    assignments.sort(
        key=lambda node: (
            getattr(node, "lineno", -1),
            getattr(node, "col_offset", -1),
        )
    )

    latest = {}
    for node in assignments:
        assignment_info = describe_assignment_node(node)
        for target_name in assignment_info["target_names"]:
            latest[target_name] = {
                "kind": assignment_info["kind"],
                "source_name": assignment_info["source_name"],
                "lineno": assignment_info["lineno"],
            }
    return latest


def describe_assignment_node(node):
    """Describe assignment node."""
    if isinstance(node, ast.Assign):
        target_names = []
        for target in node.targets:
            target_names.extend(extract_plain_target_names(target))
        source_name = extract_plain_name(node.value)
        return {
            "target_names": target_names,
            "kind": "plain_name" if source_name is not None else "other",
            "source_name": source_name,
            "lineno": node.lineno,
        }

    if isinstance(node, ast.AnnAssign):
        target_name = extract_plain_name(node.target)
        source_name = extract_plain_name(node.value)
        return {
            "target_names": [target_name] if target_name is not None else [],
            "kind": "plain_name" if source_name is not None else "other",
            "source_name": source_name,
            "lineno": node.lineno,
        }

    if isinstance(node, ast.AugAssign):
        target_name = extract_plain_name(node.target)
        return {
            "target_names": [target_name] if target_name is not None else [],
            "kind": "other",
            "source_name": None,
            "lineno": node.lineno,
        }

    return {
        "target_names": [],
        "kind": "other",
        "source_name": None,
        "lineno": getattr(node, "lineno", None),
    }


def iter_relevant_function_nodes(root_node):
    stack = [root_node]
    while stack:
        node = stack.pop()
        yield node

        for child in reversed(list(ast.iter_child_nodes(node))):
            if child is not root_node and isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
            ):
                continue
            stack.append(child)


def extract_plain_name(node):
    """Extract plain name."""
    if isinstance(node, ast.Name):
        return node.id
    return None


def extract_plain_target_names(node):
    """Extract plain target names."""
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        result = []
        for item in node.elts:
            result.extend(extract_plain_target_names(item))
        return result
    return []


def resolve_connected_forward_inputs(var_name, producer_path, forward_input_names):
    """Resolve connected forward inputs."""
    connected = []
    path_names = {item["var_name"] for item in producer_path}
    for input_name in forward_input_names:
        if input_name == var_name or input_name in path_names:
            connected.append(input_name)
    return connected


def infer_input_connection_type(
    var_name,
    connected_forward_inputs,
    direct_input_aliases,
    latest_assignments=None,
    producer_calls=None,
):
    """Infer input connection type."""
    if is_effective_direct_input_alias(
        var_name=var_name,
        direct_input_aliases=direct_input_aliases,
        latest_assignments=latest_assignments,
        producer_calls=producer_calls,
    ):
        return "direct_alias"
    if connected_forward_inputs:
        return "transformed_input"
    return "none"


def is_effective_direct_input_alias(
    var_name,
    direct_input_aliases,
    latest_assignments=None,
    producer_calls=None,
):
    if var_name not in direct_input_aliases:
        return False

    if producer_calls:
        return False

    assignment_info = (latest_assignments or {}).get(var_name)
    if assignment_info is None:
        return True

    if assignment_info.get("kind") != "plain_name":
        return False

    source_name = assignment_info.get("source_name")
    return source_name in direct_input_aliases


def resolve_shape_from_initial_inputs(
    var_name,
    connected_forward_inputs,
    input_connection_type,
    forward_input_shape_map,
):
    """Resolve shape from initial inputs."""
    if var_name in forward_input_shape_map:
        return forward_input_shape_map[var_name]

    if input_connection_type != "direct_alias":
        return None

    for input_name in connected_forward_inputs:
        shape = forward_input_shape_map.get(input_name)
        if shape is not None:
            return shape
    return None


def find_graph_node(visitor, var_name, lineno):
    """Find graph node."""
    best_prior_node = None

    for node in visitor.graph.keys():
        if node.name == var_name and node.lineno == lineno:
            return node
        if node.name != var_name or node.lineno > lineno:
            continue
        if best_prior_node is None or (node.lineno, node.start_index) > (
            best_prior_node.lineno,
            best_prior_node.start_index,
        ):
            best_prior_node = node

    if best_prior_node is not None:
        return best_prior_node

    best_prior_node = None
    for store_dict in visitor.store:
        for _, nodes in store_dict.items():
            for node in nodes:
                if node.name == var_name and node.lineno == lineno:
                    return node
                if node.name != var_name or node.lineno > lineno:
                    continue
                if best_prior_node is None or (node.lineno, node.start_index) > (
                    best_prior_node.lineno,
                    best_prior_node.start_index,
                ):
                    best_prior_node = node

    if best_prior_node is not None:
        return best_prior_node

    return visitor.var(var_name, lineno, 0)

def extract_producer_path(visitor, var_name, merge_lineno):
    """Extract producer path."""
    variable = find_graph_node(visitor, var_name, merge_lineno)
    dependencies = visitor.query_data_dependency(variable)
    if not dependencies:
        dependencies = [variable]

    ordered = sorted(
        dependencies,
        key=lambda item: (item.lineno, item.start_index),
        reverse=True,
    )
    result = []
    seen = set()
    for item in ordered:
        key = (item.name, item.lineno, item.start_index)
        if key in seen:
            continue
        seen.add(key)
        result.append({"var_name": item.name, "line_no": item.lineno})
    return result


def extract_producer_calls(
    visitor,
    producer_path,
    init_definitions,
    traced_shapes=None,
    forward_input_shape_map=None,
):
    """Extract producer calls."""
    result = []
    seen = set()
    traced_shapes = traced_shapes or {}
    forward_input_shape_map = forward_input_shape_map or {}

    for item in producer_path:
        line_no = item["line_no"]
        var_name = item["var_name"]

        direct_layer_name = extract_module_name_from_func_key(var_name)
        if direct_layer_name is not None and direct_layer_name in init_definitions:
            dedup_key = (direct_layer_name, line_no)
            if dedup_key not in seen:
                seen.add(dedup_key)
                init_definition = init_definitions[direct_layer_name]
                matched_func_key = find_matching_function_key(visitor, direct_layer_name, line_no)
                result.append(
                    build_structured_producer_call(
                        visitor=visitor,
                        layer_name=direct_layer_name,
                        api_name=init_definition.module_type,
                        call_line_no=line_no,
                        init_line_no=init_definition.line_no,
                        end_line_no=init_definition.end_line_no,
                        original_call=init_definition.code_line,
                        output_var_name=resolve_output_var_name(visitor, matched_func_key, fallback=None),
                        func_key=matched_func_key,
                        traced_shapes=traced_shapes,
                        forward_input_shape_map=forward_input_shape_map,
                    )
                )

        function_calls = visitor.lineno_function_call.get(line_no, [])
        for func_key in function_calls:
            returned_vars = visitor.func_key_return_value.get(func_key, [])
            if not call_returns_var(returned_vars, var_name, line_no):
                continue

            layer_name = extract_module_name_from_func_key(func_key.name)
            if layer_name is None or layer_name not in init_definitions:
                functional_call = resolve_functional_producer_call(
                    visitor,
                    func_key,
                    var_name,
                    line_no,
                    traced_shapes=traced_shapes,
                    forward_input_shape_map=forward_input_shape_map,
                )
                if functional_call is None:
                    continue

                dedup_key = (
                    functional_call["layer_name"],
                    functional_call["call_line_no"],
                    functional_call["api_name"],
                )
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                result.append(functional_call)
                continue

            dedup_key = (layer_name, line_no)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            init_definition = init_definitions[layer_name]
            result.append(
                build_structured_producer_call(
                    visitor=visitor,
                    layer_name=layer_name,
                    api_name=init_definition.module_type,
                    call_line_no=line_no,
                    init_line_no=init_definition.line_no,
                    end_line_no=init_definition.end_line_no,
                    original_call=init_definition.code_line,
                    output_var_name=var_name,
                    func_key=func_key,
                    traced_shapes=traced_shapes,
                    forward_input_shape_map=forward_input_shape_map,
                )
            )
    return result


def build_structured_producer_call(
    visitor,
    layer_name,
    api_name,
    call_line_no,
    init_line_no,
    original_call,
    output_var_name,
    traced_shapes,
    forward_input_shape_map,
    end_line_no=None,
    func_key=None,
):
    """Build structured producer call."""
    matched_func_key = func_key or find_matching_function_key(visitor, layer_name, call_line_no)
    input_var_names = []
    if matched_func_key is not None:
        input_var_names = extract_input_var_names_for_call(visitor, matched_func_key)

    input_shapes = [
        resolve_input_shape_for_var(
            traced_shapes=traced_shapes,
            var_name=input_var_name,
            call_line_no=call_line_no,
            forward_input_shape_map=forward_input_shape_map,
        )
        for input_var_name in input_var_names
    ]
    output_shape = resolve_output_shape_for_var(
        traced_shapes=traced_shapes,
        var_name=output_var_name,
        call_line_no=call_line_no,
        forward_input_shape_map=forward_input_shape_map,
    )

    return {
        "layer_name": layer_name,
        "api_name": api_name,
        "call_line_no": call_line_no,
        "init_line_no": init_line_no,
        "end_line_no": end_line_no,
        "original_call": original_call,
        "input_var_names": input_var_names,
        "input_shapes": input_shapes,
        "input_shape": input_shapes[0] if len(input_shapes) == 1 else None,
        "output_var_name": output_var_name,
        "output_shape": output_shape,
    }


def find_matching_function_key(visitor, layer_name, line_no):
    """Find matching function key."""
    expected_name = f"self.{layer_name}"
    for func_key in visitor.lineno_function_call.get(line_no, []):
        if func_key.name == expected_name:
            return func_key
    return None


def extract_input_var_names_for_call(visitor, func_key):
    """Extract input var names for call."""
    result = []
    seen = set()
    for variable in visitor.func_key_var_params.get(func_key, []):
        if variable.name in seen:
            continue
        seen.add(variable.name)
        result.append(variable.name)
    return result


def resolve_output_var_name(visitor, func_key, fallback=None):
    """Resolve output var name."""
    if func_key is None:
        return fallback
    returned_vars = visitor.func_key_return_value.get(func_key, [])
    if returned_vars:
        return returned_vars[0].name
    return fallback


def resolve_input_shape_for_var(
    traced_shapes,
    var_name,
    call_line_no,
    forward_input_shape_map,
):
    """Resolve input shape for var."""
    shape = resolve_shape_for_var_before_line(
        traced_shapes=traced_shapes,
        var_name=var_name,
        line_no=call_line_no,
    )
    if shape is not None:
        return shape
    return forward_input_shape_map.get(var_name)


def resolve_output_shape_for_var(
    traced_shapes,
    var_name,
    call_line_no,
    forward_input_shape_map,
):
    """Resolve output shape for var."""
    shape = resolve_shape_for_var_at_or_before_line(
        traced_shapes=traced_shapes,
        var_name=var_name,
        line_no=call_line_no,
    )
    if shape is not None:
        return shape
    return forward_input_shape_map.get(var_name)


def resolve_shape_for_var_before_line(traced_shapes, var_name, line_no):
    """Resolve shape for var before line."""
    for lineno in sorted(traced_shapes.keys(), reverse=True):
        if lineno >= line_no:
            continue
        for item in traced_shapes[lineno]:
            if item.get("var_name") == var_name and item.get("shape") is not None:
                return item.get("shape")
    return None


def resolve_shape_for_var_at_or_before_line(traced_shapes, var_name, line_no):
    """Resolve shape for var at or before line."""
    for lineno in sorted(traced_shapes.keys(), reverse=True):
        if lineno > line_no:
            continue
        for item in traced_shapes[lineno]:
            if item.get("var_name") == var_name and item.get("shape") is not None:
                return item.get("shape")
    return None


def resolve_functional_producer_call(
    visitor,
    func_key,
    var_name,
    line_no,
    traced_shapes=None,
    forward_input_shape_map=None,
):
    """Resolve functional producer call."""
    if func_key.name != "[call]":
        return None

    constructor_key = find_functional_constructor_key(visitor, func_key)
    if constructor_key is None:
        return None

    return build_structured_producer_call(
        visitor=visitor,
        layer_name=var_name,
        api_name=constructor_key.name,
        call_line_no=line_no,
        init_line_no=constructor_key.lineno,
        original_call=visitor.source_lines[line_no - 1].strip(),
        output_var_name=var_name,
        func_key=func_key,
        traced_shapes=traced_shapes or {},
        forward_input_shape_map=forward_input_shape_map or {},
    )


def find_functional_constructor_key(visitor, outer_call_key):
    """Find functional constructor key."""
    params = visitor.func_key_var_params.get(outer_call_key, [])
    nested_call_keys = [item for item in params if is_recorded_function_key(visitor, item)]
    if not nested_call_keys:
        return None

    for call_key in nested_call_keys:
        if call_key.name != "[call]":
            return call_key

    for call_key in nested_call_keys:
        resolved = find_functional_constructor_key(visitor, call_key)
        if resolved is not None:
            return resolved

    return None


def is_recorded_function_key(visitor, item):
    return item in visitor.func_key_raw_params or item in visitor.func_key_var_params


def call_returns_var(returned_vars, var_name, line_no):
    for variable in returned_vars:
        if variable.name == var_name and variable.lineno == line_no:
            return True
    return False


def extract_module_name_from_func_key(func_name):
    """Extract module name from func key."""
    if not func_name.startswith("self."):
        return None
    parts = func_name.split(".")
    if len(parts) != 2:
        return None
    return parts[1]


def resolve_shape_before_merge(traced_shapes, var_name, merge_lineno):
    """Resolve shape before merge."""
    for lineno in sorted(traced_shapes.keys(), reverse=True):
        if lineno >= merge_lineno:
            continue
        for item in traced_shapes[lineno]:
            if item.get("var_name") == var_name:
                return item.get("shape")
    return None


def fill_missing_shapes_from_error_message(branches, runtime_error_info):
    """Fill missing shapes from error message."""
    shapes = extract_shapes_from_error_message(runtime_error_info.get("error_message", ""))
    if len(shapes) < 2:
        return

    missing_indexes = []
    for index, branch in enumerate(branches):
        if branch["shape"] is None:
            missing_indexes.append(index)

    if not missing_indexes:
        return

    if len(shapes) != len(branches):
        if len(missing_indexes) == 1:
            branches[missing_indexes[0]]["shape"] = shapes[-1]
        return

    for index, shape in enumerate(shapes):
        if branches[index]["shape"] is None:
            branches[index]["shape"] = shape


def extract_shapes_from_error_message(error_message):
    """Extract shapes from error message."""
    received_shapes = extract_received_shapes_literal(error_message)
    if received_shapes:
        return received_shapes

    parsed_collection = extract_shape_collection_literal(error_message)
    if parsed_collection:
        return parsed_collection

    shapes = []
    stack = []
    start = None

    for index, char in enumerate(error_message):
        if char in "([":
            if not stack:
                start = index
            stack.append(char)
            continue

        if char in ")]" and stack:
            stack.pop()
            if not stack and start is not None:
                raw_shape = error_message[start:index + 1]
                parsed = parse_shape_literal(raw_shape)
                if parsed is not None:
                    shapes.append(parsed)
                start = None
    return shapes


def extract_received_shapes_literal(error_message):
    """Extract received shapes literal."""
    pattern = re.compile(
        r"Received shapes\s+(\([^()\n]*\)|\[[^\[\]\n]*\])\s+and\s+(\([^()\n]*\)|\[[^\[\]\n]*\])"
    )
    match = pattern.search(error_message)
    if match is None:
        return []

    shapes = []
    for raw_shape in match.groups():
        parsed = parse_shape_literal(raw_shape)
        if parsed is None:
            return []
        shapes.append(parsed)
    return shapes


def extract_shape_collection_literal(error_message):
    """Extract shape collection literal."""
    anchor = error_message.find("input_shape=")
    if anchor == -1:
        return []

    start = error_message.find("[", anchor)
    if start == -1:
        return []

    stack = []
    for index in range(start, len(error_message)):
        char = error_message[index]
        if char == "[":
            stack.append(char)
            continue

        if char == "]" and stack:
            stack.pop()
            if not stack:
                raw_collection = error_message[start:index + 1]
                return parse_shape_collection_literal(raw_collection)

    return []


def parse_shape_collection_literal(raw_collection):
    """Parse shape collection literal."""
    try:
        value = ast.literal_eval(raw_collection)
    except (ValueError, SyntaxError):
        return []

    if not isinstance(value, (list, tuple)):
        return []

    shapes = []
    for item in value:
        parsed = parse_shape_literal(repr(item))
        if parsed is not None:
            shapes.append(parsed)
    return shapes


def parse_shape_literal(raw_shape):
    """Parse shape literal."""
    try:
        value = ast.literal_eval(raw_shape)
    except (ValueError, SyntaxError):
        return None

    if not isinstance(value, (list, tuple)):
        return None

    shape = []
    for item in value:
        if item is None:
            shape.append(None)
            continue
        if not isinstance(item, int):
            return None
        shape.append(item)

    if len(shape) < 2:
        return None
    return shape


def annotate_branch_roles(branches, forward_input_names=None):
    """Annotate branch roles."""
    forward_input_names = set(forward_input_names or [])

    for branch in branches:
        if branch.get("is_direct_input_alias"):
            branch["role"] = "inputs"
            continue

        var_name = branch["var_name"]
        lowered_name = var_name.lower()
        if var_name in forward_input_names:
            branch["role"] = "inputs"
            continue
        if "inputs" in lowered_name or "input" in lowered_name:
            branch["role"] = "inputs"

    non_inputs = [branch for branch in branches if branch["role"] != "inputs"]
    if len(non_inputs) == 1:
        non_inputs[0]["role"] = "main-path"
        return
    if len(non_inputs) != 2:
        return

    explicit_branch = None
    for branch in non_inputs:
        if "branch" in branch["var_name"]:
            explicit_branch = branch
            break

    if explicit_branch is not None:
        explicit_branch["role"] = "branch-path"
        for branch in non_inputs:
            if branch is not explicit_branch:
                branch["role"] = "main-path"
        return

    branch_path = min(non_inputs, key=lambda item: item["path_length"])
    branch_path["role"] = "branch-path"
    for branch in non_inputs:
        if branch is not branch_path:
            branch["role"] = "main-path"
