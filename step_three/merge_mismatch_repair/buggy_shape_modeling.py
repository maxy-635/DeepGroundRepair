from typing import Any


class BuggyShapeModeler:
    """Implement the buggy shape modeler component."""

    def __init__(self, step2_result: dict[str, Any]) -> None:
        """Initialize the instance."""
        self.step2_result = step2_result

        root_cause = step2_result.get("root_cause") or {}
        self.merge_context = root_cause.get("merge_context") or {}
        self.merge_annotation_hints = root_cause.get("merge_annotation_hints") or []
        self.target_layers = root_cause.get("target_layers") or []
        self.modeling_sequence = root_cause.get("target_layer_modeling_sequence") or []

    def hint_matches_target_layer(
        self,
        hint: dict[str, Any],
        target_layer: dict[str, Any],
        target_layer_index: int | None = None,
    ) -> bool:
        if isinstance(target_layer_index, int) and hint.get("target_layer_index") == target_layer_index:
            return True

        target_layer_line = target_layer.get("line_no")
        if isinstance(target_layer_line, int) and hint.get("target_layer_line_no") == target_layer_line:
            return True

        layer_name = target_layer.get("layer_name")
        target_call_line = target_layer.get("call_line_no")
        return bool(
            layer_name
            and hint.get("layer_name") == layer_name
            and hint.get("line_no") == target_call_line
        )

    def extract_input_tensor_shape(self, target_layer: dict[str, Any]) -> list[int] | None:
        """Extract input tensor shape."""
        direct_input_shape = target_layer.get("input_shape")
        if direct_input_shape is not None:
            return list(direct_input_shape)

        layer_name = target_layer.get("layer_name")
        init_line_no = target_layer.get("line_no")
        call_line_no = target_layer.get("call_line_no")
        merge_context = self.merge_context
        forward_input_shapes = merge_context.get("forward_input_shapes") or {}

        for branch in merge_context.get("branches") or []:
            producer_calls = branch.get("producer_calls") or []
            for index, call_info in enumerate(producer_calls):
                same_layer_name = layer_name and call_info.get("layer_name") == layer_name
                same_init_line = call_info.get("init_line_no") == init_line_no
                same_call_line = call_info.get("call_line_no") == call_line_no

                if not (same_layer_name or same_init_line or same_call_line):
                    continue

                structured_input_shape = call_info.get("input_shape")
                if structured_input_shape is not None:
                    return list(structured_input_shape)

                structured_input_shapes = call_info.get("input_shapes") or []
                if len(structured_input_shapes) == 1 and structured_input_shapes[0] is not None:
                    return list(structured_input_shapes[0])

                for producer_item in branch.get("producer_path") or []:
                    input_var_name = producer_item.get("var_name")
                    if (
                        producer_item.get("line_no") == call_info.get("call_line_no")
                        and input_var_name in forward_input_shapes
                    ):
                        return list(forward_input_shapes[input_var_name])

                next_call = producer_calls[index + 1] if index + 1 < len(producer_calls) else None
                if next_call and next_call.get("output_shape") is not None:
                    return list(next_call["output_shape"])

                connected_forward_inputs = branch.get("connected_forward_inputs") or []
                for input_name in connected_forward_inputs:
                    input_shape = forward_input_shapes.get(input_name)
                    if input_shape is not None and index == len(producer_calls) - 1:
                        return list(input_shape)

                branch_shape = branch.get("shape")
                if branch_shape is not None:
                    return list(branch_shape)

        return None

    def extract_target_output_tensor_shape(self,
        target_layer: dict[str, Any],
        primary_merge_annotation_hint_index: int | None = None,
        target_layer_index: int | None = None,
    ) -> list[int] | None:
        """Extract target output tensor shape."""
        merge_annotation_hints = self.merge_annotation_hints
        target_call_line = target_layer.get("call_line_no")

        if (
            isinstance(primary_merge_annotation_hint_index, int)
            and 0 <= primary_merge_annotation_hint_index < len(merge_annotation_hints)
        ):
            primary_hint = merge_annotation_hints[primary_merge_annotation_hint_index]
            if (
                primary_hint.get("oracle_shape") is not None
                and self.hint_matches_target_layer(
                    hint=primary_hint,
                    target_layer=target_layer,
                    target_layer_index=target_layer_index,
                )
            ):
                return list(primary_hint["oracle_shape"])

        for hint in merge_annotation_hints or []:
            if (
                hint.get("oracle_shape") is not None
                and self.hint_matches_target_layer(
                    hint=hint,
                    target_layer=target_layer,
                    target_layer_index=target_layer_index,
                )
            ):
                return list(hint["oracle_shape"])

        same_line_hints = [
            hint
            for hint in merge_annotation_hints or []
            if hint.get("line_no") == target_call_line and hint.get("oracle_shape") is not None
        ]
        if len(same_line_hints) == 1:
            return list(same_line_hints[0]["oracle_shape"])
        if same_line_hints and not any(
            hint.get("target_layer_index") is not None
            or hint.get("target_layer_line_no") is not None
            or hint.get("layer_name") is not None
            for hint in same_line_hints
        ):
            return list(same_line_hints[0]["oracle_shape"])

        return None

    def extract_target_layer_io_shape_pairs(self) -> list[dict[str, Any]]:
        """Extract target layer io shape pairs."""
        target_layers = self.target_layers
        modeling_sequence = self.modeling_sequence

        shape_pairs = []
        for item in modeling_sequence:
            target_layer_index = item.get("target_layer_index")
            if target_layer_index is None or target_layer_index >= len(target_layers):
                continue

            target_layer = target_layers[target_layer_index]
            shape_pairs.append(
                {
                    "target_layer_index": target_layer_index,
                    "layer_name": item.get("layer_name") or target_layer.get("layer_name"),
                    "modeled_input_shape": item.get("modeled_input_shape"),
                    "oracle_output_shape": item.get("oracle_output_shape"),
                }
            )

        if shape_pairs:
            return shape_pairs

        for index, target_layer in enumerate(target_layers):
            shape_pairs.append(
                {
                    "target_layer_index": index,
                    "layer_name": target_layer.get("layer_name"),
                    "modeled_input_shape": self.extract_input_tensor_shape(
                        target_layer=target_layer,
                    ),
                    "oracle_output_shape": self.extract_target_output_tensor_shape(
                        target_layer=target_layer,
                        target_layer_index=index,
                    ),
                }
            )
        return shape_pairs

