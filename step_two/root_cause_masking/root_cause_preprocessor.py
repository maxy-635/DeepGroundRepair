import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class RewriteItem:
    """Implement the rewrite item component."""

    line_no: int
    end_line_no: int | None = None
    original_call: str | None = None
    rewritten_call: str | None = None
    fallback_call: str | None = None
    params_to_mask: list[str] | None = None
    keep_inline_comment: bool = True


class ForwardLocalCallSimplifier:
    """Implement the forward local call simplifier component."""

    def extract_call_node(self, stmt):
        """Extract call node."""
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            return stmt.value
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.value, ast.Call):
            return stmt.value
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            return stmt.value
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
            return stmt.value
        return None

    def mask_call_argument_values(self, call_node: ast.Call):
        """Mask call argument values."""
        call_node.args = [
            ast.Constant(value="mask")
            for _ in call_node.args
        ]
        call_node.keywords = [
            ast.keyword(arg=keyword.arg, value=ast.Constant(value="mask"))
            for keyword in call_node.keywords
        ]

    def mask_shape_transform_call(self, call_node: ast.Call):
        """Mask shape transform call."""
        func = call_node.func

        if isinstance(func, ast.Attribute) and func.attr in {"view", "reshape"}:
            self.mask_call_argument_values(call_node)
            return True

        if isinstance(func, ast.Name) and func.id == "reshape":
            self.mask_call_argument_values(call_node)
            return True

        return False

    def simplify(self, code_line: str):
        try:
            tree = ast.parse(code_line)
        except SyntaxError:
            return code_line.strip()

        if not tree.body:
            return code_line.strip()

        stmt = tree.body[0]
        call_node = self.extract_call_node(stmt)
        if call_node is None:
            return code_line.strip()

        if not self.mask_shape_transform_call(call_node):
            return code_line.strip()

        return ast.unparse(stmt)


class RootCauseRewriteCollector:
    """Implement the root cause rewrite collector component."""

    def __init__(self, forward_call_simplifier=None):
        self.forward_call_simplifier = (
            forward_call_simplifier or ForwardLocalCallSimplifier()
        )

    def collect(self, root_cause):
        target_layers = root_cause.get("target_layers")
        if target_layers:
            return [
                RewriteItem(
                    line_no=layer["line_no"],
                    end_line_no=layer.get("end_line_no"),
                    original_call=layer.get("original_call"),
                    rewritten_call=(
                        layer.get("masked_api_call")
                        if layer.get("params_to_mask")
                        else None
                    ),
                    fallback_call=layer.get("masked_api_call"),
                    keep_inline_comment=False,
                )
                for layer in target_layers
            ]

        init_definition = root_cause.get("init_definition") or {}
        if "line_no" in init_definition and "original_call" in init_definition:
            return [
                RewriteItem(
                    line_no=init_definition["line_no"],
                    end_line_no=init_definition.get("end_line_no"),
                    original_call=init_definition.get("original_call"),
                    fallback_call=init_definition.get("masked_api_call"),
                    params_to_mask=init_definition.get("params_to_mask"),
                    keep_inline_comment=False,
                )
            ]

        forward_definition = root_cause.get("forward_definition") or {}
        if "line_no" in forward_definition and "original_call" in forward_definition:
            rewritten_call = self.forward_call_simplifier.simplify(
                forward_definition["original_call"]
            )
            if rewritten_call is not None:
                return [
                    RewriteItem(
                        line_no=forward_definition["line_no"],
                        original_call=forward_definition["original_call"],
                        rewritten_call=rewritten_call,
                        keep_inline_comment=True,
                    )
                ]

        return []
