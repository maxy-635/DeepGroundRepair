import io
import tokenize


class MergeAnnotationHelper:
    """Implement the merge annotation helper component."""

    NORMAL_SHAPE_COMMENT_PREFIX = "# output tensor shape:"
    MERGE_ORACLE_COMMENT_PREFIX = "# output tensor shape of this layer should be "

    def __init__(self, shape_formatter):
        """Initialize the instance."""
        self.shape_formatter = shape_formatter

    def build_merge_oracle_comment(self, merge_annotation_hint):
        """Build merge oracle comment."""
        if not merge_annotation_hint:
            return None

        oracle_shape = merge_annotation_hint.get("oracle_shape")
        if oracle_shape is None:
            return None

        return self.MERGE_ORACLE_COMMENT_PREFIX + self.shape_formatter(oracle_shape)

    def get_merge_annotation_hints(self, root_cause):
        """Return merge annotation hints."""
        if not root_cause or root_cause.get("root_cause_type") != "Tensor_Merge_Mismatch":
            return []

        hints = root_cause.get("merge_annotation_hints") or []
        normalized = [hint for hint in hints if isinstance(hint.get("line_no"), int)]
        normalized.sort(key=lambda item: item["line_no"])
        return normalized

    def split_code_and_comment(self, original_line):
        """Split code and comment."""
        has_newline = original_line.endswith("\n")
        code_part = original_line.rstrip("\n")
        comment_part = None

        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(original_line).readline))
            for token in tokens:
                if token.type == tokenize.COMMENT:
                    code_part = original_line[:token.start[1]].rstrip()
                    comment_part = original_line[token.start[1] :].rstrip("\n")
                    break
        except tokenize.TokenError:
            code_part = original_line.rstrip("\n")
            comment_part = None

        return code_part, comment_part, has_newline

    def replace_line_comment(self, lines, lineno, comment_text):
        """Replace line comment."""
        if lineno < 1 or lineno > len(lines):
            return

        code_part, _, has_newline = self.split_code_and_comment(lines[lineno - 1])
        if not code_part.strip():
            return

        updated_line = f"{code_part}  {comment_text}"
        if has_newline:
            updated_line += "\n"
        lines[lineno - 1] = updated_line

    def clear_line_comment(self, lines, lineno, expected_comment_prefix=None):
        """Clear line comment."""
        if lineno < 1 or lineno > len(lines):
            return

        code_part, comment_part, has_newline = self.split_code_and_comment(lines[lineno - 1])
        if not code_part.strip() or not comment_part:
            return

        if expected_comment_prefix and not comment_part.startswith(expected_comment_prefix):
            return

        updated_line = code_part
        if has_newline:
            updated_line += "\n"
        lines[lineno - 1] = updated_line

    def collect_target_branch_cleanup_lines(self, root_cause, merge_annotation_hints):
        """Collect target branch cleanup lines."""
        if not root_cause or root_cause.get("root_cause_type") != "Tensor_Merge_Mismatch":
            return set()
        if not merge_annotation_hints:
            return set()

        first_oracle_line = min(
            hint["line_no"]
            for hint in merge_annotation_hints
            if isinstance(hint.get("line_no"), int)
        )
        oracle_lines = {
            hint["line_no"]
            for hint in merge_annotation_hints
            if isinstance(hint.get("line_no"), int)
        }

        merge_context = root_cause.get("merge_context") or {}
        branches = merge_context.get("branches") or []
        target_branch_index = root_cause.get("target_branch_index")
        if not isinstance(target_branch_index, int):
            return set()
        if target_branch_index < 0 or target_branch_index >= len(branches):
            return set()

        target_branch = branches[target_branch_index]
        merge_lineno = merge_context.get("merge_lineno")
        candidate_lines = set()

        for item in target_branch.get("producer_path") or []:
            line_no = item.get("line_no")
            if isinstance(line_no, int):
                candidate_lines.add(line_no)

        for call_info in target_branch.get("producer_calls") or []:
            line_no = call_info.get("call_line_no")
            if isinstance(line_no, int):
                candidate_lines.add(line_no)

        cleanup_lines = set()
        for line_no in candidate_lines:
            if line_no <= first_oracle_line:
                continue
            if line_no in oracle_lines:
                continue
            if isinstance(merge_lineno, int) and line_no >= merge_lineno:
                continue
            cleanup_lines.add(line_no)

        return cleanup_lines

    def apply_merge_oracle_annotation(self, lines, root_cause, insertions, adjust_lineno_fn):
        """Apply merge oracle annotation."""
        merge_annotation_hints = self.get_merge_annotation_hints(root_cause)
        hints_by_line = {}
        for merge_annotation_hint in merge_annotation_hints:
            line_no = merge_annotation_hint.get("line_no")
            if not isinstance(line_no, int):
                continue
            hints_by_line.setdefault(line_no, []).append(merge_annotation_hint)

        for line_no, line_hints in hints_by_line.items():
            merge_annotation_hint = self.select_merge_annotation_hint_for_line(line_hints)
            comment_text = self.build_merge_oracle_comment(merge_annotation_hint)
            if comment_text is None:
                continue

            adjusted_lineno = adjust_lineno_fn(line_no, insertions or [])
            self.replace_line_comment(lines, adjusted_lineno, comment_text)

    def select_merge_annotation_hint_for_line(self, line_hints):
        """Select merge annotation hint for line."""
        if not line_hints:
            return None

        indexed_hints = list(enumerate(line_hints))
        _, selected_hint = max(
            indexed_hints,
            key=lambda item: (
                item[1].get("target_layer_line_no")
                if isinstance(item[1].get("target_layer_line_no"), int)
                else -1,
                item[0],
            ),
        )
        return selected_hint

    def clear_target_branch_trailing_shape_annotations(
        self,
        lines,
        root_cause,
        insertions,
        adjust_lineno_fn,
    ):
        """Clear target branch trailing shape annotations."""
        merge_annotation_hints = self.get_merge_annotation_hints(root_cause)
        cleanup_lines = self.collect_target_branch_cleanup_lines(
            root_cause=root_cause,
            merge_annotation_hints=merge_annotation_hints,
        )
        for line_no in sorted(cleanup_lines):
            adjusted_lineno = adjust_lineno_fn(line_no, insertions or [])
            self.clear_line_comment(
                lines,
                adjusted_lineno,
                expected_comment_prefix=self.NORMAL_SHAPE_COMMENT_PREFIX,
            )
