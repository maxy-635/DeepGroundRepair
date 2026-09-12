import ast
import io
import tokenize


class CommentCleaner:
    """Implement the comment cleaner component."""
    METHOD_NAME_MAP = {
        "pytorch": "forward",
        "paddlepaddle": "forward",
        "tensorflow": "call",
    }

    def __init__(self, dll_type: str):
        normalized_type = dll_type.strip().lower()
        if normalized_type not in self.METHOD_NAME_MAP:
            raise ValueError(
                f"Unsupported dll_type: {dll_type}. Expected one of: {', '.join(self.METHOD_NAME_MAP)}"
            )
        self.dll_type = normalized_type
        self.target_method = self.METHOD_NAME_MAP[normalized_type]

    def find_model_method_range(self, source_code: str):
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == self.target_method:
                return node.lineno, node.end_lineno
        return None, None

    def _remove_inline_comments_from_block(self, block: str) -> str:
        """Remove inline comments from block."""
        lines = block.splitlines(keepends=True)
        result_lines = lines[:]

        for tok in tokenize.generate_tokens(io.StringIO(block).readline):
            if tok.type != tokenize.COMMENT:
                continue

            line_idx = tok.start[0] - 1
            comment_col = tok.start[1]
            original_line = result_lines[line_idx]
            prefix = original_line[:comment_col]

            if prefix.strip() == "":
                continue

            has_newline = original_line.endswith("\n")
            new_line = prefix.rstrip()
            if has_newline:
                new_line += "\n"
            result_lines[line_idx] = new_line

        return "".join(result_lines)

    def remove_comments_in_target_method(self, source_code: str) -> str:
        start_line, end_line = self.find_model_method_range(source_code)
        if start_line is None:
            return source_code

        lines = source_code.splitlines(keepends=True)

        method_block = "".join(lines[start_line - 1:end_line])
        cleaned_block = self._remove_inline_comments_from_block(method_block)
        lines[start_line - 1:end_line] = [cleaned_block]

        return "".join(lines)

