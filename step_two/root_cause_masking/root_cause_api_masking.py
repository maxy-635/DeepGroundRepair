import io
import tokenize

from step_two.root_cause_masking.api_call_rewriter import ApiCallRewriter
from step_two.root_cause_masking.root_cause_preprocessor import (
    RewriteItem,
    RootCauseRewriteCollector,
)


class SourceLineMasker:
    """Implement the source line masker component."""

    def apply(self, lines, rewrite_item: RewriteItem):

        line_no = rewrite_item.line_no
        end_line_no = self.resolve_end_line_no(lines, rewrite_item)
        rewritten_call = rewrite_item.rewritten_call

        if line_no < 1 or line_no > len(lines):
            return
        if end_line_no < line_no:
            end_line_no = line_no

        original_line = lines[line_no - 1]
        original_segment = lines[line_no - 1:end_line_no]
        has_newline = bool(original_segment) and original_segment[-1].endswith("\n")
        indent = original_line[: len(original_line) - len(original_line.lstrip())]

        inline_comment, code_part = self.extract_inline_comment(original_line)
        has_trailing_comma = code_part.rstrip().endswith(",")

        updated_line = self.apply_indent(rewritten_call, indent)
        if has_trailing_comma and not rewritten_call.rstrip().endswith(","):
            updated_line += ","
        if rewrite_item.keep_inline_comment and inline_comment:
            updated_line += f"  {inline_comment}"
        if has_newline:
            updated_line += "\n"

        updated_lines = updated_line.splitlines(keepends=True)
        updated_lines = self.pad_to_original_line_count(
            updated_lines=updated_lines,
            original_segment=original_segment,
        )
        lines[line_no - 1:end_line_no] = updated_lines

    def resolve_end_line_no(self, lines, rewrite_item: RewriteItem):
        """Resolve end line no."""
        end_line_no = rewrite_item.end_line_no
        if end_line_no is None and rewrite_item.original_call:
            end_line_no = rewrite_item.line_no + len(rewrite_item.original_call.splitlines()) - 1
        if end_line_no is None:
            end_line_no = rewrite_item.line_no
        return min(end_line_no, len(lines))

    def apply_indent(self, rewritten_call, indent):
        """Apply indent."""
        rewritten_call = rewritten_call.rstrip("\n")
        if not rewritten_call:
            return indent
        first_line, *rest_lines = rewritten_call.splitlines()
        if first_line.startswith((" ", "\t")):
            return "\n".join([first_line, *rest_lines])
        return "\n".join([f"{indent}{first_line}", *rest_lines])

    def pad_to_original_line_count(self, updated_lines, original_segment):
        """Pad to original line count."""
        missing_line_count = len(original_segment) - len(updated_lines)
        if missing_line_count <= 0:
            return updated_lines

        newline = "\n" if original_segment[-1].endswith("\n") else ""
        return updated_lines + [newline for _ in range(missing_line_count)]

    def extract_inline_comment(self, original_line):
        """Extract inline comment."""
        inline_comment = ""
        code_part = original_line.rstrip("\n")
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(original_line).readline))
            for token in tokens:
                if token.type == tokenize.COMMENT:
                    inline_comment = original_line[token.start[1] :].rstrip("\n")
                    code_part = original_line[: token.start[1]].rstrip()
                    break
        except tokenize.TokenError:
            inline_comment = ""

        return inline_comment, code_part


class RootCauseApiMasker:
    """Implement the root cause api masker component."""

    def __init__(self, rewrite_collector=None, source_line_masker=None, api_call_rewriter=None):
        self.rewrite_collector = rewrite_collector or RootCauseRewriteCollector()
        self.source_line_masker = source_line_masker or SourceLineMasker()
        self.api_call_rewriter = api_call_rewriter or ApiCallRewriter()

    def mask(self, origin_code: str, root_cause):
        if root_cause is None:
            return origin_code

        lines = origin_code.splitlines(keepends=True)
        rewrite_items = self.rewrite_collector.collect(root_cause)
        if not rewrite_items:
            return origin_code

        for rewrite_item in sorted(rewrite_items, key=lambda item: item.line_no):
            resolved_item = self.resolve_rewrite_item(origin_code, rewrite_item)
            if resolved_item.rewritten_call is None:
                continue
            self.source_line_masker.apply(lines, resolved_item)

        return "".join(lines)

    def resolve_rewrite_item(self, origin_code: str, rewrite_item: RewriteItem):
        """Resolve one rewrite item into a concrete rewritten call string."""
        if rewrite_item.rewritten_call is not None:
            return rewrite_item

        if rewrite_item.original_call is None:
            return rewrite_item

        rewritten_call = self.api_call_rewriter.rewrite_masked_call(
            source_code=origin_code,
            code_line=rewrite_item.original_call,
            mask_parameter_names=rewrite_item.params_to_mask,
        )

        if rewritten_call == rewrite_item.original_call and rewrite_item.fallback_call:
            rewritten_call = rewrite_item.fallback_call

        return RewriteItem(
            line_no=rewrite_item.line_no,
            end_line_no=rewrite_item.end_line_no,
            original_call=rewrite_item.original_call,
            rewritten_call=rewritten_call,
            fallback_call=rewrite_item.fallback_call,
            params_to_mask=rewrite_item.params_to_mask,
            keep_inline_comment=rewrite_item.keep_inline_comment,
        )
