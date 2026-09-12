import ast
from typing import Optional
from step_two.shape_annotation.comment_cleaner import CommentCleaner
from step_two.shape_annotation.merge_annotation_utils import MergeAnnotationHelper
from step_two.tensor_shape_trace.tensor_shape_tracer import TensorShapeTrace


class ShapeAnnotation:
    """Implement the shape annotation component."""


    ENTRY_METHOD_NAMES = {"forward", "call"}

    def __init__(self):
        self.trace_result = None
        self.merge_annotation_helper = MergeAnnotationHelper(self.format_shape)

    def format_shape(self, shape):
        """Format shape."""
        if isinstance(shape, (list, tuple)):
            return "(" + ", ".join(str(dim) for dim in shape) + ")"
        return str(shape)

    def build_shape_comment(self, shape_items):
        """Build shape comment."""
        formatted_shapes = []
        for item in shape_items:
            shape = item.get("shape")
            if shape is None:
                continue
            formatted_shapes.append(self.format_shape(shape))

        if not formatted_shapes:
            return None
        return "# output tensor shape: " + "; ".join(formatted_shapes)

    def build_initial_input_comments(self, shape_items, indent):
        """Build initial input comments."""
        comments = []
        for item in shape_items:
            var_name = item.get("var_name")
            shape = item.get("shape")
            if var_name is None or shape is None:
                continue
            comments.append(
                f"{indent}# the initial shape of {var_name} is: {self.format_shape(shape)}\n"
            )
        return comments

    def insert_initial_input_comments(
        self,
        lines,
        cleaned_code: str,
        initial_input_shapes: Optional[dict] = None,
    ):
        insertions = []
        if not initial_input_shapes:
            return insertions

        root = ast.parse(cleaned_code)

        for node in ast.walk(root):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in self.ENTRY_METHOD_NAMES or not node.body:
                continue

            shape_items = initial_input_shapes.get(node.lineno)
            if not shape_items:
                continue

            first_stmt = node.body[0]
            insert_at = node.lineno
            indent = " " * first_stmt.col_offset
            comments = self.build_initial_input_comments(shape_items, indent)
            if comments:
                insertions.append((insert_at, comments))

        for insert_at, comments in reversed(insertions):
            lines[insert_at:insert_at] = comments

        return [
            {
                "after_lineno": insert_at,
                "count": len(comments),
            }
            for insert_at, comments in insertions
        ]

    def adjust_lineno_after_insertions(self, lineno, insertions):
        adjusted_lineno = lineno
        for insertion in insertions:
            after_lineno = insertion["after_lineno"]
            count = insertion["count"]
            if lineno > after_lineno:
                adjusted_lineno += count
        return adjusted_lineno

    def should_skip_annotation_line(self, lines, lineno, shape_items):
        if not isinstance(lineno, int):
            return True
        if lineno < 1 or lineno > len(lines):
            return True
        if not shape_items:
            return True

        original_line = lines[lineno - 1]
        stripped_line = original_line.strip()
        if not stripped_line:
            return True
        if stripped_line.startswith("#"):
            return True
        if "#" in original_line:
            return True

        return False

    def annotate_output_shapes(self, lines, traced_shapes: dict, insertions=None):
        """Annotate output shapes."""
        for lineno, shape_items in traced_shapes.items():
            adjusted_lineno = self.adjust_lineno_after_insertions(
                lineno,
                insertions or [],
            )

            if self.should_skip_annotation_line(lines, adjusted_lineno, shape_items):
                continue

            original_line = lines[adjusted_lineno - 1]
            shape_comment = self.build_shape_comment(shape_items)
            if shape_comment is None:
                continue

            has_newline = original_line.endswith("\n")
            line_body = original_line[:-1] if has_newline else original_line
            updated_line = f"{line_body}  {shape_comment}"
            if has_newline:
                updated_line += "\n"

            lines[adjusted_lineno - 1] = updated_line

    def apply_merge_oracle_annotation(self, lines, root_cause, insertions=None):
        """Apply merge oracle annotation."""
        self.merge_annotation_helper.apply_merge_oracle_annotation(
            lines=lines,
            root_cause=root_cause,
            insertions=insertions,
            adjust_lineno_fn=self.adjust_lineno_after_insertions,
        )

    def clear_target_branch_trailing_shape_annotations(self, lines, root_cause, insertions=None):
        """Clear target branch trailing shape annotations."""
        self.merge_annotation_helper.clear_target_branch_trailing_shape_annotations(
            lines=lines,
            root_cause=root_cause,
            insertions=insertions,
            adjust_lineno_fn=self.adjust_lineno_after_insertions,
        )

    def annotation(
        self,
        cleaned_code: str,
        traced_shapes: dict,
        initial_input_shapes: Optional[dict] = None,
        root_cause: Optional[dict] = None,
    ):
        lines = cleaned_code.splitlines(keepends=True)
        insertions = self.insert_initial_input_comments(
            lines,
            cleaned_code,
            initial_input_shapes=initial_input_shapes,
        )
        self.annotate_output_shapes(lines, traced_shapes, insertions=insertions)
        self.apply_merge_oracle_annotation(lines, root_cause, insertions=insertions)
        self.clear_target_branch_trailing_shape_annotations(
            lines,
            root_cause,
            insertions=insertions,
        )
        return "".join(lines)
    
    def main(self, source_pyfile: str, target_pyfile: str, dll_type: str):
        """Run the main workflow."""

        with open(source_pyfile, "r", encoding="utf-8") as f:
            source = f.read()

        comment_cleaner = CommentCleaner(dll_type)
        cleaned_code = comment_cleaner.remove_comments_in_target_method(source)

        shape_tracer = TensorShapeTrace(python_code=cleaned_code)
        self.trace_result = shape_tracer.main()
        traced_shapes = self.trace_result["traced_shapes"]
        initial_input_shapes = self.trace_result.get("initial_input_shapes", {})

        annotated_code = self.annotation(
            cleaned_code,
            traced_shapes,
            initial_input_shapes=initial_input_shapes,
        )
        with open(target_pyfile, "w", encoding="utf-8") as f:
            f.write(annotated_code)

        print(f"Saved annotated file to: {target_pyfile}")
