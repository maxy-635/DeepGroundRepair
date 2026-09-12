import ast
import inspect
from dataclasses import dataclass

from step_two.root_cause_masking.api_call_rewriter_utils import (
    RewriteNotApplicable,
    build_rewritten_keywords,
    collect_import_references,
    extract_target_call,
    has_unsupported_unpacking,
    map_positional_arguments_to_parameters,
    resolve_runtime_callable,
    validate_keyword_arguments,
)


MASK_VALUE = "mask"


@dataclass(frozen=True)
class ApiCallRewriteResult:
    """Implement the api call rewrite result component."""

    original_code: str
    keyword_call: str
    masked_call: str
    changed: bool
    resolved_callable: str | None = None
    failure_reason: str | None = None


class ApiCallRewriter:
    """Implement the api call rewriter component."""

    def __init__(self, mask_value: str = MASK_VALUE):
        self.mask_value = mask_value

    def rewrite(self, source_code: str, code_line: str, mask_parameter_names=None):
        original_code = code_line.strip()
        if not original_code:
            return ApiCallRewriteResult(
                original_code=original_code,
                keyword_call=original_code,
                masked_call=original_code,
                changed=False,
                failure_reason="empty_code",
            )

        import_references = collect_import_references(source_code)
        resolved_cache = {}
        mask_parameter_names = (
            set(mask_parameter_names)
            if mask_parameter_names is not None
            else None
        )

        try:
            keyword_call, resolved_callable = self._rewrite_statement(
                code_line=original_code,
                import_references=import_references,
                resolved_cache=resolved_cache,
                mask_values=False,
                mask_parameter_names=None,
            )
            masked_call, _ = self._rewrite_statement(
                code_line=original_code,
                import_references=import_references,
                resolved_cache=resolved_cache,
                mask_values=True,
                mask_parameter_names=mask_parameter_names,
            )
        except RewriteNotApplicable as exc:
            return ApiCallRewriteResult(
                original_code=original_code,
                keyword_call=original_code,
                masked_call=original_code,
                changed=False,
                resolved_callable=exc.resolved_callable,
                failure_reason=exc.reason,
            )

        return ApiCallRewriteResult(
            original_code=original_code,
            keyword_call=keyword_call,
            masked_call=masked_call,
            changed=masked_call != original_code,
            resolved_callable=resolved_callable,
        )

    def rewrite_masked_call(self, source_code: str, code_line: str, mask_parameter_names=None):
        """Rewrite masked call."""
        return self.rewrite(
            source_code=source_code,
            code_line=code_line,
            mask_parameter_names=mask_parameter_names,
        ).masked_call

    def _rewrite_statement(
        self,
        code_line: str,
        import_references,
        resolved_cache,
        mask_values: bool,
        mask_parameter_names=None,
    ):
        """Rewrite statement."""
        try:
            tree = ast.parse(code_line)
        except SyntaxError as exc:
            raise RewriteNotApplicable("syntax_error") from exc

        if not tree.body:
            raise RewriteNotApplicable("empty_statement")

        stmt = tree.body[0]
        target_call = extract_target_call(stmt)
        if target_call is None:
            raise RewriteNotApplicable("unsupported_statement")

        if has_unsupported_unpacking(target_call):
            raise RewriteNotApplicable("contains_unpacking")

        resolved_callable, resolved_name = resolve_runtime_callable(
            target_call.func,
            import_references=import_references,
            resolved_modules=resolved_cache,
        )
        if resolved_callable is None:
            raise RewriteNotApplicable(
                "call_target_unresolved",
                resolved_callable=resolved_name,
            )

        try:
            signature = inspect.signature(resolved_callable)
        except (TypeError, ValueError) as exc:
            raise RewriteNotApplicable(
                "signature_unavailable",
                resolved_callable=resolved_name,
            ) from exc

        positional_param_names = map_positional_arguments_to_parameters(
            call_node=target_call,
            signature=signature,
        )
        validate_keyword_arguments(target_call, signature)

        rewritten_keywords = build_rewritten_keywords(
            call_node=target_call,
            positional_param_names=positional_param_names,
            mask_values=mask_values,
            mask_value=self.mask_value,
            mask_parameter_names=mask_parameter_names,
        )
        target_call.args = []
        target_call.keywords = rewritten_keywords
        ast.fix_missing_locations(tree)
        return ast.unparse(stmt), resolved_name
