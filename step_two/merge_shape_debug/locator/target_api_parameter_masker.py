import ast
import inspect
from step_two.root_cause_masking.api_call_rewriter import ApiCallRewriter
from step_two.root_cause_masking.api_call_rewriter_utils import (
    collect_import_references,
    resolve_runtime_callable,
)
from step_two.merge_shape_debug.locator.target_api_locator_utils import (
    get_called_api_name,
    unique_list,
)


MASK_VALUE = "mask"


class TargetApiParameterMasker:
    """Implement the target api parameter masker component."""

    def __init__(self, api_parameter_map, api_call_rewriter=None):
        """Initialize the instance."""
        self.api_parameter_map = api_parameter_map or {}
        self.api_call_rewriter = api_call_rewriter or ApiCallRewriter()

    def get_target_params(self, framework, mismatch_type, semantic_type):
        """Return target params."""
        framework_rules = self.api_parameter_map.get(framework or "", {})
        mismatch_rules = framework_rules.get(mismatch_type, {})
        rule = mismatch_rules.get(semantic_type, {})
        return {
            "param_names": unique_list(rule.get("param_names", [])),
        }

    def find_constructor_call_node(self, tree, api_name):
        """Find constructor call node."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and get_called_api_name(node) == api_name:
                return node
        return None

    def normalize_positional_arguments(self, original_call, source_code=None):
        """Normalize positional arguments."""
        if not source_code:
            return original_call

        rewrite_result = self.api_call_rewriter.rewrite(
            source_code=source_code,
            code_line=original_call,
        )
        if rewrite_result.failure_reason is None:
            return rewrite_result.keyword_call

        return original_call

    def resolve_signature_parameter_names(self, call_node, source_code=None):
        """Resolve signature parameter names."""
        if not source_code:
            return None

        import_references = collect_import_references(source_code)
        resolved_callable, _ = resolve_runtime_callable(
            call_node.func,
            import_references=import_references,
            resolved_modules={},
        )
        if resolved_callable is None:
            return None

        try:
            signature = inspect.signature(resolved_callable)
        except (TypeError, ValueError):
            return None

        return set(signature.parameters.keys())

    def can_append_missing_parameter(self, param_name, signature_param_names):
        if signature_param_names is None:
            return False
        return param_name in signature_param_names

    def append_missing_mask_keywords(self, call_node, concrete_params, source_code=None):
        """Append missing mask keywords."""
        signature_param_names = self.resolve_signature_parameter_names(
            call_node=call_node,
            source_code=source_code,
        )

        existing_keyword_names = {
            keyword.arg
            for keyword in call_node.keywords
            if keyword.arg is not None
        }
        appended_param_names = []
        for param_name in concrete_params or []:
            if param_name in existing_keyword_names:
                continue
            if not self.can_append_missing_parameter(
                param_name=param_name,
                signature_param_names=signature_param_names,
            ):
                continue

            call_node.keywords.append(
                ast.keyword(
                    arg=param_name,
                    value=ast.Constant(value=MASK_VALUE),
                )
            )
            appended_param_names.append(param_name)

        return appended_param_names

    def mask_existing_argument_values(self, original_call, api_name, source_code=None):
        """Mask existing argument values."""
        if source_code:
            rewrite_result = self.api_call_rewriter.rewrite(
                source_code=source_code,
                code_line=original_call,
                mask_parameter_names=None,
            )
            if rewrite_result.failure_reason is None:
                return rewrite_result.masked_call

        try:
            tree = ast.parse(original_call)
        except SyntaxError:
            return original_call.strip()

        call_node = self.find_constructor_call_node(tree, api_name)
        if call_node is None:
            return original_call.strip()

        call_node.args = [
            ast.copy_location(ast.Constant(value=MASK_VALUE), arg)
            for arg in call_node.args
        ]
        call_node.keywords = [
            ast.keyword(
                arg=keyword.arg,
                value=ast.copy_location(ast.Constant(value=MASK_VALUE), keyword.value),
            )
            for keyword in call_node.keywords
        ]
        ast.fix_missing_locations(tree)
        return ast.unparse(tree.body[0])

    def mask_keyword_params(self, original_call, api_name, concrete_params, source_code=None):
        """Mask keyword params."""
        normalized_call = self.normalize_positional_arguments(
            original_call=original_call,
            source_code=source_code,
        )
        try:
            tree = ast.parse(normalized_call)
        except SyntaxError:
            return (
                self.mask_existing_argument_values(original_call, api_name, source_code=source_code),
                [],
                "all_argument_value_fallback",
            )

        call_node = self.find_constructor_call_node(tree, api_name)
        if call_node is None:
            return (
                self.mask_existing_argument_values(original_call, api_name, source_code=source_code),
                [],
                "all_argument_value_fallback",
            )

        concrete_param_set = set(concrete_params or [])
        masked_param_names = []
        for keyword in call_node.keywords:
            if keyword.arg in concrete_param_set:
                keyword.value = ast.copy_location(
                    ast.Constant(value=MASK_VALUE),
                    keyword.value,
                )
                masked_param_names.append(keyword.arg)

        missing_masked_param_names = self.append_missing_mask_keywords(
            call_node=call_node,
            concrete_params=concrete_params,
            source_code=source_code,
        )
        masked_param_names.extend(missing_masked_param_names)

        if not masked_param_names:
            return (
                self.mask_existing_argument_values(original_call, api_name, source_code=source_code),
                [],
                "all_argument_value_fallback",
            )

        ast.fix_missing_locations(tree)
        return (
            ast.unparse(tree.body[0]),
            unique_list(masked_param_names),
            "semantic_keyword_parameter",
        )
