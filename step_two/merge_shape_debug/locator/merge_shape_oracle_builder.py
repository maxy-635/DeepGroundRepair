from step_two.merge_shape_debug.analysis.tensor_shape_normalizer import TensorShapeNormalizer


class MergeShapeOracleBuilder:
    """Implement the merge shape oracle builder component."""

    def __init__(self):
        """Initialize the instance."""
        self.shape_normalizer = TensorShapeNormalizer()

    def build_merge_annotation_hint(self, primary_layer, target_branch, merge_context):
        """Build merge annotation hint."""
        annotation_line_no = self.find_target_layer_call_line(
            primary_layer=primary_layer,
            target_branch=target_branch,
            merge_context=merge_context,
        )
        oracle_shape = self.select_oracle_shape(target_branch, merge_context)
        if annotation_line_no is None or oracle_shape is None:
            return None

        return {
            "line_no": annotation_line_no,
            "oracle_shape": oracle_shape,
            "layer_name": primary_layer.get("layer_name"),
            "target_layer_line_no": primary_layer.get("line_no"),
            "call_line_no": primary_layer.get("call_line_no"),
        }

    def build_merge_annotation_hints(
        self,
        target_layers,
        target_branch,
        merge_context,
        traced_shapes,
    ):
        """Build merge annotation hints."""
        base_oracle_shape = self.select_oracle_shape(target_branch, merge_context)
        if base_oracle_shape is None:
            return []

        ordered_layers = self.sort_target_layers_for_annotation(
            target_layers=target_layers,
            target_branch=target_branch,
            merge_context=merge_context,
        )
        if not ordered_layers:
            return []

        current_oracle_shape = list(base_oracle_shape)
        target_layer_indexes = {
            id(layer): index
            for index, layer in enumerate(target_layers or [])
        }
        hints = []
        for layer in ordered_layers:
            annotation_line_no = self.find_target_layer_call_line(
                primary_layer=layer,
                target_branch=target_branch,
                merge_context=merge_context,
            )
            if annotation_line_no is None:
                continue

            actual_output_shape = self.resolve_layer_output_shape(
                layer=layer,
                target_branch=target_branch,
                merge_context=merge_context,
                traced_shapes=traced_shapes,
            )
            layer_oracle_shape = self.compose_layer_oracle_shape(
                layer=layer,
                current_oracle_shape=current_oracle_shape,
                base_oracle_shape=base_oracle_shape,
                actual_output_shape=actual_output_shape,
                merge_context=merge_context,
            )
            if layer_oracle_shape is None:
                continue

            hints.append(
                {
                    "line_no": annotation_line_no,
                    "oracle_shape": layer_oracle_shape,
                    "target_layer_index": target_layer_indexes.get(id(layer)),
                    "layer_name": layer.get("layer_name"),
                    "target_layer_line_no": layer.get("line_no"),
                    "call_line_no": layer.get("call_line_no"),
                    "mismatch_types": list(layer.get("mismatch_types") or []),
                }
            )
            current_oracle_shape = list(layer_oracle_shape)

        return hints

    def select_primary_merge_annotation_hint(
        self,
        primary_layer,
        target_branch,
        merge_context,
        merge_annotation_hints,
    ):
        """Select primary merge annotation hint."""
        primary_line_no = self.find_target_layer_call_line(
            primary_layer=primary_layer,
            target_branch=target_branch,
            merge_context=merge_context,
        )
        for hint in merge_annotation_hints or []:
            same_target_layer_line = (
                hint.get("target_layer_line_no") is not None
                and hint.get("target_layer_line_no") == primary_layer.get("line_no")
            )
            same_layer_name = (
                hint.get("layer_name") is not None
                and hint.get("layer_name") == primary_layer.get("layer_name")
            )
            same_call_line = hint.get("line_no") == primary_line_no
            if same_target_layer_line or (same_layer_name and same_call_line):
                return hint

        return self.build_merge_annotation_hint(
            primary_layer=primary_layer,
            target_branch=target_branch,
            merge_context=merge_context,
        )

    def sort_target_layers_for_annotation(self, target_layers, target_branch, merge_context):
        """Sort target layers for annotation."""
        sortable_layers = []
        for layer in target_layers or []:
            annotation_line_no = self.find_target_layer_call_line(
                primary_layer=layer,
                target_branch=target_branch,
                merge_context=merge_context,
            )
            sortable_layers.append(
                (
                    annotation_line_no if isinstance(annotation_line_no, int) else float("inf"),
                    layer.get("line_no", float("inf")),
                    layer,
                )
            )

        sortable_layers.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in sortable_layers]

    def resolve_layer_output_shape(self, layer, target_branch, merge_context, traced_shapes):
        """Resolve layer output shape."""
        call_line_no = layer.get("call_line_no")
        if isinstance(call_line_no, int):
            traced_items = traced_shapes.get(call_line_no) or []
            resolved_shape = self.pick_shape_from_traced_items(traced_items)
            if resolved_shape is not None:
                return resolved_shape

        branch_shape = target_branch.get("shape")
        if branch_shape is None:
            return None

        merge_lineno = merge_context.get("merge_lineno")
        if (
            isinstance(call_line_no, int)
            and isinstance(merge_lineno, int)
            and call_line_no < merge_lineno
        ):
            return list(branch_shape)
        return None

    def pick_shape_from_traced_items(self, traced_items):
        if not traced_items:
            return None

        for item in reversed(traced_items):
            shape = item.get("shape")
            if shape is not None:
                return list(shape)
        return None

    def compose_layer_oracle_shape(
        self,
        layer,
        current_oracle_shape,
        base_oracle_shape,
        actual_output_shape,
        merge_context,
    ):
        """Compose layer oracle shape."""
        if current_oracle_shape is None:
            return None

        mismatch_types = set(layer.get("mismatch_types") or [])
        layer_oracle_shape = self.compose_shape_by_mismatch_types(
            template_shape=actual_output_shape,
            reference_shape=current_oracle_shape,
            mismatch_types=mismatch_types,
            merge_context=merge_context,
        )
        if layer_oracle_shape is None:
            layer_oracle_shape = list(current_oracle_shape)

        return layer_oracle_shape

    def compose_shape_by_mismatch_types(
        self,
        template_shape,
        reference_shape,
        mismatch_types,
        merge_context,
    ):
        """Compose shape by mismatch types."""
        if reference_shape is None:
            return None
        if template_shape is None:
            return list(reference_shape)

        result = list(template_shape)
        mismatch_types = set(mismatch_types or [])

        if "spatial" in mismatch_types:
            result = self.replace_spatial_dims(
                current_shape=result,
                source_shape=reference_shape,
                merge_context=merge_context,
            )
        if "channel" in mismatch_types:
            result = self.replace_channel_dim(
                current_shape=result,
                source_shape=reference_shape,
                merge_context=merge_context,
            )

        if not (mismatch_types & {"spatial", "channel"}):
            result = self.project_shape_onto_template(
                source_shape=reference_shape,
                template_shape=template_shape,
                merge_context=merge_context,
            )

        return self.replace_relaxed_merge_dim(
            current_shape=result,
            source_shape=template_shape,
            merge_context=merge_context,
        )

    def project_shape_onto_template(self, source_shape, template_shape, merge_context):
        """Project shape onto template."""
        if template_shape is None:
            return list(source_shape) if source_shape is not None else None
        if source_shape is None:
            return list(template_shape)
        if len(source_shape) == len(template_shape):
            return self.replace_relaxed_merge_dim(
                current_shape=source_shape,
                source_shape=template_shape,
                merge_context=merge_context,
            )

        projected_shape = list(template_shape)
        projected_shape = self.replace_channel_dim(
            current_shape=projected_shape,
            source_shape=source_shape,
            merge_context=merge_context,
        )
        return projected_shape

    def replace_relaxed_merge_dim(self, current_shape, source_shape, merge_context):
        """Replace relaxed merge dim."""
        if current_shape is None or source_shape is None:
            return current_shape
        if len(current_shape) != len(source_shape):
            return list(current_shape)

        kind = str((merge_context or {}).get("merge_kind", "")).lower()
        if kind not in {"cat", "concat", "concatenate"}:
            return list(current_shape)

        if self.shape_normalizer.get_feature_shape(current_shape) is None:
            return list(current_shape)

        relaxed_feature_axis = self.shape_normalizer.resolve_relaxed_feature_axis(
            current_shape,
            merge_context,
        )
        relaxed_index = self.resolve_original_axis_index(
            shape=current_shape,
            feature_axis_index=relaxed_feature_axis,
        )
        if relaxed_index is None or relaxed_index >= len(source_shape):
            return list(current_shape)

        result = list(current_shape)
        result[relaxed_index] = source_shape[relaxed_index]
        return result

    def replace_channel_dim(self, current_shape, source_shape, merge_context):
        """Replace channel dim."""
        if current_shape is None or source_shape is None:
            return current_shape

        current_meta = self.shape_normalizer.describe_feature_axes(
            current_shape,
            merge_context,
        )
        source_meta = self.shape_normalizer.describe_feature_axes(
            source_shape,
            merge_context,
        )
        if current_meta is None or source_meta is None:
            return current_shape

        current_index = self.resolve_original_axis_index(
            shape=current_shape,
            feature_axis_index=current_meta["channel_index"],
        )
        source_index = self.resolve_original_axis_index(
            shape=source_shape,
            feature_axis_index=source_meta["channel_index"],
        )
        if current_index is None or source_index is None:
            return current_shape

        result = list(current_shape)
        result[current_index] = source_shape[source_index]
        return result

    def replace_spatial_dims(self, current_shape, source_shape, merge_context):
        """Replace spatial dims."""
        if current_shape is None or source_shape is None:
            return current_shape

        current_meta = self.shape_normalizer.describe_feature_axes(
            current_shape,
            merge_context,
        )
        source_meta = self.shape_normalizer.describe_feature_axes(
            source_shape,
            merge_context,
        )
        if current_meta is None or source_meta is None:
            return current_shape

        current_spatial_indexes = current_meta.get("spatial_indexes") or []
        source_spatial_indexes = source_meta.get("spatial_indexes") or []
        if len(current_spatial_indexes) != len(source_spatial_indexes):
            return current_shape

        result = list(current_shape)
        for current_feature_index, source_feature_index in zip(
            current_spatial_indexes,
            source_spatial_indexes,
        ):
            current_index = self.resolve_original_axis_index(
                shape=current_shape,
                feature_axis_index=current_feature_index,
            )
            source_index = self.resolve_original_axis_index(
                shape=source_shape,
                feature_axis_index=source_feature_index,
            )
            if current_index is None or source_index is None:
                continue
            result[current_index] = source_shape[source_index]

        return result

    def resolve_original_axis_index(self, shape, feature_axis_index):
        """Resolve original axis index."""
        if shape is None or feature_axis_index is None:
            return None

        feature_shape = self.shape_normalizer.get_feature_shape(shape)
        if feature_shape is None:
            return None

        if len(shape) == len(feature_shape) + 1:
            return feature_axis_index + 1
        if feature_axis_index >= len(shape):
            return None
        return feature_axis_index

    def find_target_layer_call_line(self, primary_layer, target_branch, merge_context):
        """Find target layer call line."""
        if not primary_layer:
            return None

        merge_lineno = merge_context.get("merge_lineno")
        target_call_line_no = primary_layer.get("call_line_no")
        if self.is_valid_annotation_line(target_call_line_no, merge_lineno):
            return target_call_line_no

        producer_calls = target_branch.get("producer_calls") or []
        if not producer_calls:
            return None

        matched_lines = []
        for call_info in producer_calls:
            call_line_no = call_info.get("call_line_no")
            if not self.is_valid_annotation_line(call_line_no, merge_lineno):
                continue

            if call_info.get("layer_name") == primary_layer.get("layer_name"):
                matched_lines.append(call_line_no)
                continue

            primary_init_line_no = primary_layer.get("line_no")
            call_init_line_no = call_info.get("init_line_no")
            if (
                isinstance(primary_init_line_no, int)
                and isinstance(call_init_line_no, int)
                and primary_init_line_no == call_init_line_no
            ):
                matched_lines.append(call_line_no)

        if not matched_lines:
            return None

        return max(matched_lines)

    def is_valid_annotation_line(self, call_line_no, merge_lineno):
        if not isinstance(call_line_no, int):
            return False
        if isinstance(merge_lineno, int) and call_line_no >= merge_lineno:
            return False
        return True

    def select_oracle_shape(self, target_branch, merge_context):
        """Select oracle shape."""
        branches = merge_context.get("branches") or []
        shape_counts = {}
        shape_order = []

        for branch in branches:
            if branch is target_branch:
                continue

            shape = branch.get("shape")
            if shape is None:
                continue

            shape_key = tuple(shape) if isinstance(shape, list) else tuple(shape)
            if shape_key not in shape_counts:
                shape_counts[shape_key] = 0
                shape_order.append(shape_key)
            shape_counts[shape_key] += 1

        if not shape_counts:
            return None

        best_shape_key = max(
            shape_order,
            key=lambda key: (shape_counts[key], -shape_order.index(key)),
        )
        return list(best_shape_key)
