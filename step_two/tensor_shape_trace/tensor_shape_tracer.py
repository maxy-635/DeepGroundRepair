import ast
import traceback
from step_two.tensor_shape_trace.ast_parser import Visitor
from step_two.tensor_shape_trace.runtime_shapes_collector import RuntimeShapesCollector
from step_two.tensor_shape_trace.sequential_runtime_failure_tracker import SequentialRuntimeFailureTracker


class TensorShapeTrace:
    """Implement the tensor shape trace component."""
    def __init__(self, python_code: str, filename: str = "<memory_code>"):
        """Initialize the instance."""
        if not python_code:
            raise ValueError("python_code must not be empty.")
        self.python_code = python_code
        self.filename = filename
        self.sequential_failure_tracker = SequentialRuntimeFailureTracker()

    def build_ast(self):
        """Build ast."""
        root_node = ast.parse(self.python_code, self.filename)
        source_lines = self.python_code.splitlines(keepends=True)
        visitor = Visitor(source_lines)
        visitor.visit(root_node)
        return visitor

    def init_runtime_trace(self, visitor):
        """Initialize runtime trace."""
        collector = RuntimeShapesCollector()
        traced_filename = collector.default_filter_filename(self.filename)
        collector.reset_shapes_collection()
        collector.init_shapes_collection(
            filenames=[traced_filename],
            lineno_varname=visitor.lineno_varname,
            trace_shape=True,
        )
        return collector

    def execute_and_trace(self, collector):
        """Execute and trace."""
        namespace = {"__name__": "__main__"}
        runtime_error = None
        traced_shapes = {}
        runtime_error_info = None
        traceback_text = None

        try:
            with collector.collect():
                with self.sequential_failure_tracker.track():
                    exec(compile(self.python_code, self.filename, "exec"), namespace, namespace)
        except Exception as exception:
            runtime_error = exception
            traceback_text = traceback.format_exc()

        finally:
            traced = collector.dumps_stats()
            initial_input_shapes = collector.dumps_initial_inputs()
            normalized_traced_shapes = {}
            for lineno in sorted(traced):
                normalized_traced_shapes[lineno] = []
                for item in traced[lineno]:
                    shape_info = {
                        "var_name": item["var_name"],
                        "shape": item["shape"],
                    }
                    normalized_traced_shapes[lineno].append(shape_info)

            traced_shapes = normalized_traced_shapes

            if runtime_error is not None:
                runtime_error_info = {
                    "error_type": type(runtime_error).__name__,
                    "error_message": str(runtime_error),
                    "traceback": traceback_text,
                    "container_failures": self.sequential_failure_tracker.dumps_failures(),
                }

            collector.stop_shapes_collection()

        return {
            "traced_shapes": traced_shapes,
            "initial_input_shapes": initial_input_shapes,
            "runtime_error_info": runtime_error_info,
        }

    def main(self):
        """Run the main workflow."""
        visitor = self.build_ast()
        collector = self.init_runtime_trace(visitor)
        trace_result = self.execute_and_trace(collector)

        return trace_result

