from step_two.crash_locator.root_cause_locator import InitModuleFinder
from step_two.merge_shape_debug.analysis.shape_compare import ShapeCompare
from step_two.merge_shape_debug.context.merge_ast_utils import parse_merge_crash_code
from step_two.merge_shape_debug.context.merge_trace_utils import build_merge_context
from step_two.merge_shape_debug.context.merge_trace_utils import build_visitor


class MergeMismatchDetector:
    """Implement the merge mismatch detector component."""

    def __init__(self):
        """Initialize the instance."""
        self.shape_compare = ShapeCompare()

    def collect_init_definitions(self, origin_code):
        """Collect init definitions."""
        if not origin_code:
            return {}
        return InitModuleFinder(origin_code).collect()

    def is_branch_merge_point(
        self,
        runtime_error_info,
        crash_location,
        origin_code,
        traced_shapes,
        initial_input_shapes,
        source_path,
        init_definitions,
    ):
        if not origin_code or not source_path:
            return False

        visitor = build_visitor(origin_code, source_path)
        merge_context = build_merge_context(
            visitor=visitor,
            traced_shapes=traced_shapes or {},
            initial_input_shapes=initial_input_shapes or {},
            crash_location=crash_location,
            init_definitions=init_definitions,
            runtime_error_info=runtime_error_info,
            source_code=origin_code,
        )
        if merge_context is None:
            return False

        branches = merge_context.get("branches", [])
        if len(branches) < 2:
            return False

        producer_signatures = []
        for branch in branches:
            signature = self.get_branch_signature(branch)
            if signature is not None:
                producer_signatures.append(signature)

        if len(set(producer_signatures)) >= 2:
            return True

        var_names = [
            branch.get("var_name")
            for branch in branches
            if branch.get("var_name") is not None
        ]
        return len(set(var_names)) >= 2

    def get_branch_signature(self, branch):
        """Return branch signature."""
        producer_path = branch.get("producer_path") or []
        if producer_path:
            return tuple(
                (item.get("var_name"), item.get("line_no"))
                for item in producer_path
            )

        producer_calls = branch.get("producer_calls") or []
        if producer_calls:
            return tuple(
                (item.get("layer_name"), item.get("call_line_no"), item.get("init_line_no"))
                for item in producer_calls
            )

        var_name = branch.get("var_name")
        if var_name is not None:
            return ("var", var_name)
        return None

    def has_shape_mismatch_signal(self, error_message):
        mismatch_keywords = [
            "incompatible shapes",
            "incompatible shape",
            "dimensions must be equal",
            "shape mismatch",
            "matching shapes",
            "must have the same shape",
            "sizes of tensors must match",
            "except in dimension",
            "operands could not be broadcast",
            "Broadcast dimension mismatch",
            "broadcast dimension mismatch",
            "must match the size of tensor",
        ]
        lowered_message = error_message.lower()
        return any(keyword in lowered_message for keyword in mismatch_keywords)

    def detect_mismatch_types(self, merge_context):
        """Detect mismatch types."""
        branches = [branch for branch in merge_context["branches"] if branch["shape"]]
        if len(branches) < 2:
            return []

        mismatch_types = []
        if self.shape_compare.has_spatial_mismatch(branches, merge_context):
            mismatch_types.append("spatial")
        if self.shape_compare.has_channel_mismatch(branches, merge_context):
            mismatch_types.append("channel")
        return mismatch_types

    def is_merge_mismatch(self, runtime_error_info, crash_location, origin_code=None,
                        traced_shapes=None, initial_input_shapes=None, source_path=None
                    ):
        init_definitions = self.collect_init_definitions(origin_code)

        merge_info = parse_merge_crash_code(
            crash_location.crash_code,
            init_definitions,
            source_code=origin_code,
        )
        is_merge_like_op = merge_info is not None

        is_branch_merge = self.is_branch_merge_point(
            runtime_error_info=runtime_error_info,
            crash_location=crash_location,
            origin_code=origin_code,
            traced_shapes=traced_shapes,
            initial_input_shapes=initial_input_shapes,
            source_path=source_path,
            init_definitions=init_definitions,
        )

        has_error_signal = self.has_shape_mismatch_signal(
            runtime_error_info.get("error_message", "")
        )

        return any([is_merge_like_op, is_branch_merge, has_error_signal])
