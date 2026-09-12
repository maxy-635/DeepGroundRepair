import ast
import copy


MERGE_LAYER_TYPES = {"Add", "Concatenate", "Multiply"}
MERGE_FUNCTION_NAMES = {"add", "cat", "concat", "concatenate", "multiply"}


def parse_merge_crash_code(crash_code, init_definitions, source_code=None):
    """Parse merge crash code."""
    try:
        tree = ast.parse(crash_code)
    except SyntaxError:
        return None

    if not tree.body:
        return None

    stmt = tree.body[0]
    expr_node, statement_kind = extract_stmt_expression(stmt)
    if expr_node is None:
        return None

    wrapper_index = WRAPPER_EXPANSION_HELPER.build_wrapper_index(source_code)
    resolved = MERGE_CALL_RESOLVER.resolve_merge_expression(
        expr_node,
        init_definitions,
        wrapper_index,
    )
    if resolved is None:
        return None

    input_vars = []
    for node in resolved["input_nodes"]:
        var_name = extract_simple_name(node)
        if var_name is not None:
            input_vars.append(var_name)

    if len(input_vars) < 2:
        return None

    return {
        "merge_kind": resolved["merge_kind"],
        "concat_axis": resolved["concat_axis"],
        "input_vars": input_vars,
        "statement_kind": statement_kind,
    }


def extract_stmt_expression(stmt):
    """Extract stmt expression."""
    if isinstance(stmt, ast.Assign):
        value = stmt.value
        statement_kind = "assign"
    elif isinstance(stmt, ast.AugAssign):
        value = ast.BinOp(left=copy.deepcopy(stmt.target), op=stmt.op, right=stmt.value)
        statement_kind = "augassign"
    elif isinstance(stmt, ast.Expr):
        value = stmt.value
        statement_kind = "expr"
    else:
        return None, None

    if isinstance(value, (ast.Call, ast.BinOp)):
        return value, statement_kind
    return None, None



class WrapperExpansionHelper:
    """Implement the wrapper expansion helper component."""

    def __init__(self, merge_call_resolver=None):
        self.merge_call_resolver = merge_call_resolver

    def set_merge_call_resolver(self, merge_call_resolver):
        self.merge_call_resolver = merge_call_resolver

    def build_wrapper_index(self, source_code):
        """Build wrapper index."""
        if not source_code:
            return {}

        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return {}

        wrappers = {}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                wrappers[node.name] = node
                continue
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        wrappers[f"self.{item.name}"] = item
        return wrappers

    def resolve_wrapper_call(self, call_node, init_definitions, wrapper_index, depth):
        """Resolve wrapper call."""
        wrapper_name = extract_call_name(call_node.func)
        if wrapper_name is None:
            return None

        wrapper_def = wrapper_index.get(wrapper_name)
        if wrapper_def is None:
            return None

        bound_arguments = self.bind_wrapper_arguments(wrapper_def, call_node)
        for return_expr in self.iter_return_expressions(wrapper_def):
            substituted_expr = self.substitute_arguments(return_expr, bound_arguments)
            resolved = self.merge_call_resolver.resolve_merge_expression(
                substituted_expr,
                init_definitions,
                wrapper_index,
                depth=depth + 1,
            )
            if resolved is not None:
                return resolved
        return None

    def bind_wrapper_arguments(self, wrapper_def, call_node):
        """Bind wrapper arguments."""
        positional_args = list(getattr(wrapper_def.args, "posonlyargs", [])) + list(wrapper_def.args.args)
        parameters = [arg.arg for arg in positional_args]
        if parameters and parameters[0] == "self":
            parameters = parameters[1:]

        bound = {}
        for name, value in zip(parameters, call_node.args):
            bound[name] = value

        for keyword in call_node.keywords:
            if keyword.arg is not None:
                bound[keyword.arg] = keyword.value
        return bound

    def iter_return_expressions(self, wrapper_def):
        for node in ast.walk(wrapper_def):
            if isinstance(node, ast.Return) and node.value is not None:
                yield node.value

    def substitute_arguments(self, expr_node, bound_arguments):

        class ArgumentSubstituter(ast.NodeTransformer):
            def visit_Name(self, node):
                replacement = bound_arguments.get(node.id)
                if replacement is None:
                    return node
                return ast.copy_location(
                    ast.fix_missing_locations(copy.deepcopy(replacement)),
                    node,
                )

        substituted = ArgumentSubstituter().visit(ast.fix_missing_locations(copy.deepcopy(expr_node)))
        return ast.fix_missing_locations(substituted)


class MergeCallResolver:
    """Implement the merge call resolver component."""

    def __init__(self, concat_axis_resolver, wrapper_helper=None):
        self.concat_axis_resolver = concat_axis_resolver
        self.wrapper_helper = wrapper_helper

    def extract_merge_kind_from_binop(self, op):
        """Extract merge kind from binop."""
        if isinstance(op, ast.Add):
            return "add"
        if isinstance(op, ast.Mult):
            return "multiply"
        return None

    def flatten_binop_input_nodes(self, expr_node, merge_kind):
        """Flatten binop input nodes."""
        if not isinstance(expr_node, ast.BinOp):
            return [expr_node]

        current_kind = self.extract_merge_kind_from_binop(expr_node.op)
        if current_kind != merge_kind:
            return [expr_node]

        input_nodes = []
        input_nodes.extend(self.flatten_binop_input_nodes(expr_node.left, merge_kind))
        input_nodes.extend(self.flatten_binop_input_nodes(expr_node.right, merge_kind))
        return input_nodes

    def extract_merge_input_nodes(self, call_node):
        """Extract merge input nodes."""
        if not call_node.args:
            return []
        first_arg = call_node.args[0]
        if isinstance(first_arg, (ast.List, ast.Tuple)):
            return list(first_arg.elts)
        return [first_arg]

    def extract_direct_function_input_nodes(self, call_node):
        """Extract direct function input nodes."""
        if not call_node.args:
            return []
        first_arg = call_node.args[0]
        if isinstance(first_arg, (ast.List, ast.Tuple)):
            return list(first_arg.elts)
        return list(call_node.args)

    def resolve_merge_expression(self, expr_node, init_definitions, wrapper_index, depth=0):
        """Resolve merge expression."""
        if depth > 4:
            return None

        if isinstance(expr_node, ast.BinOp):
            merge_kind = self.extract_merge_kind_from_binop(expr_node.op)
            if merge_kind is None:
                return None
            return {
                "merge_kind": merge_kind,
                "concat_axis": None,
                "input_nodes": self.flatten_binop_input_nodes(expr_node, merge_kind),
            }

        if not isinstance(expr_node, ast.Call):
            return None

        resolved = self.resolve_explicit_merge_call(expr_node, init_definitions)
        if resolved is not None:
            return resolved

        return self.wrapper_helper.resolve_wrapper_call(
            expr_node,
            init_definitions,
            wrapper_index,
            depth,
        )

    def resolve_explicit_merge_call(self, call_node, init_definitions):
        """Resolve explicit merge call."""
        if isinstance(call_node.func, ast.Attribute):
            module_name = extract_self_module_name(call_node.func)
            if module_name and module_name in init_definitions:
                merge_kind = init_definitions[module_name].module_type
                if merge_kind in MERGE_LAYER_TYPES:
                    return {
                        "merge_kind": merge_kind,
                        "concat_axis": self.concat_axis_resolver.resolve_effective_concat_axis(
                            explicit_axis=extract_axis_from_call(call_node),
                            merge_kind=merge_kind,
                            module_name=module_name,
                            init_definitions=init_definitions,
                        ),
                        "input_nodes": self.extract_merge_input_nodes(call_node),
                    }

            function_name = call_node.func.attr
            if function_name in MERGE_FUNCTION_NAMES:
                return {
                    "merge_kind": function_name,
                    "concat_axis": self.concat_axis_resolver.resolve_effective_concat_axis(
                        explicit_axis=extract_axis_from_call(call_node),
                        merge_kind=function_name,
                    ),
                    "input_nodes": self.extract_direct_function_input_nodes(call_node),
                }

        if isinstance(call_node.func, ast.Name):
            function_name = call_node.func.id
            if function_name in MERGE_FUNCTION_NAMES:
                return {
                    "merge_kind": function_name,
                    "concat_axis": self.concat_axis_resolver.resolve_effective_concat_axis(
                        explicit_axis=extract_axis_from_call(call_node),
                        merge_kind=function_name,
                    ),
                    "input_nodes": self.extract_direct_function_input_nodes(call_node),
                }

        if isinstance(call_node.func, ast.Call):
            inner_call = call_node.func
            constructor_name = extract_call_name(inner_call.func)
            if constructor_name:
                layer_type = constructor_name.split(".")[-1]
                if layer_type in MERGE_LAYER_TYPES:
                    return {
                        "merge_kind": layer_type,
                        "concat_axis": self.concat_axis_resolver.resolve_effective_concat_axis(
                            explicit_axis=extract_axis_from_call(inner_call),
                            merge_kind=layer_type,
                        ),
                        "input_nodes": self.extract_merge_input_nodes(call_node),
                    }

        return None



def extract_simple_name(node):
    """Extract simple name."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = []
        value = node
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
            return ".".join(reversed(parts))
    return None



def extract_call_name(node):
    """Extract call name."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return extract_simple_name(node)
    return None



def extract_self_module_name(node):
    """Extract self module name."""
    if not isinstance(node, ast.Attribute):
        return None
    if not isinstance(node.value, ast.Name):
        return None
    if node.value.id != "self":
        return None
    return node.attr



def extract_axis_from_call(call_node):
    """Extract axis from call."""
    for keyword in call_node.keywords:
        if keyword.arg in {"axis", "dim"}:
            return literal_number(keyword.value)
    if len(call_node.args) >= 2:
        return literal_number(call_node.args[1])
    return None


class ConcatAxisResolver:
    """Implement the concat axis resolver component."""

    def extract_axis_from_init_definition(self, init_definition):
        """Extract axis from init definition."""
        if init_definition is None:
            return None

        original_line = getattr(init_definition, "code_line", None)
        if not original_line:
            return None

        try:
            tree = ast.parse(original_line)
        except SyntaxError:
            return None

        if not tree.body:
            return None

        stmt = tree.body[0]
        value = getattr(stmt, "value", None)
        if not isinstance(value, ast.Call):
            return None

        return extract_axis_from_call(value)

    def get_default_concat_axis(self, merge_kind):
        """Return default concat axis."""
        lowered_kind = str(merge_kind or "").lower()
        if lowered_kind in {"concatenate"}:
            return -1
        if lowered_kind in {"cat", "concat"}:
            return 0
        return None

    def resolve_effective_concat_axis(
        self,
        explicit_axis,
        merge_kind,
        module_name=None,
        init_definitions=None,
    ):
        """Resolve effective concat axis."""
        if explicit_axis is not None:
            return explicit_axis

        if module_name is not None and init_definitions is not None:
            init_definition = init_definitions.get(module_name)
            init_axis = self.extract_axis_from_init_definition(init_definition)
            if init_axis is not None:
                return init_axis

            if init_definition is not None:
                default_axis = self.get_default_concat_axis(init_definition.module_type)
                if default_axis is not None:
                    return default_axis

        return self.get_default_concat_axis(merge_kind)

CONCAT_AXIS_RESOLVER = ConcatAxisResolver()
WRAPPER_EXPANSION_HELPER = WrapperExpansionHelper()
MERGE_CALL_RESOLVER = MergeCallResolver(
    concat_axis_resolver=CONCAT_AXIS_RESOLVER,
    wrapper_helper=WRAPPER_EXPANSION_HELPER,
)
WRAPPER_EXPANSION_HELPER.set_merge_call_resolver(MERGE_CALL_RESOLVER)


def literal_number(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = literal_number(node.operand)
        if value is not None:
            return -value
    return None