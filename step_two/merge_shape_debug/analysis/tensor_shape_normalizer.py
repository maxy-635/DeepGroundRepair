class TensorShapeNormalizer:
    """Implement the tensor shape normalizer component."""

    def get_feature_shape(self, shape):
        """Return feature shape."""
        if shape is None:
            return None
        if len(shape) >= 4:
            return list(shape[1:])
        if len(shape) == 2:
            return [shape[-1]]
        return list(shape)

    def infer_tensor_layout(self, merge_context):
        """Infer tensor layout."""
        if merge_context is None:
            return "channels_last"

        explicit_layout_candidates = [
            merge_context.get("data_format"),
            merge_context.get("layout"),
            merge_context.get("feature_layout"),
            merge_context.get("merge_data_format"),
            merge_context.get("merge_layout"),
            merge_context.get("tensor_layout"),
        ]
        for layout_value in explicit_layout_candidates:
            normalized_layout = self.normalize_layout_value(layout_value)
            if normalized_layout is not None:
                return normalized_layout

        framework = str(merge_context.get("framework", "")).lower()
        if framework in {"pytorch", "paddle"}:
            return "channels_first"
        if framework == "tensorflow":
            return "channels_last"
        return "channels_last"

    def normalize_layout_value(self, layout_value):
        """Normalize layout value."""
        if not isinstance(layout_value, str):
            return None

        lowered = layout_value.strip().lower()
        channels_first_aliases = {
            "channels_first",
            "channel_first",
            "ncl",
            "nchw",
            "ncdhw",
            "ncw",
        }
        channels_last_aliases = {
            "channels_last",
            "channel_last",
            "nlc",
            "nhwc",
            "ndhwc",
            "nwc",
        }

        if lowered in channels_first_aliases:
            return "channels_first"
        if lowered in channels_last_aliases:
            return "channels_last"
        return None

    def describe_feature_axes(self, shape, merge_context):
        """Describe feature axes."""
        feature_shape = self.get_feature_shape(shape)
        if feature_shape is None or not feature_shape:
            return None

        if len(feature_shape) == 1:
            return {
                "feature_shape": feature_shape,
                "channel_index": 0,
                "spatial_indexes": [],
            }

        layout = self.infer_tensor_layout(merge_context)
        if layout == "channels_first":
            return {
                "feature_shape": feature_shape,
                "channel_index": 0,
                "spatial_indexes": list(range(1, len(feature_shape))),
            }

        return {
            "feature_shape": feature_shape,
            "channel_index": len(feature_shape) - 1,
            "spatial_indexes": list(range(0, len(feature_shape) - 1)),
        }

    def normalize_feature_axis(self, axis, original_rank, feature_rank):
        """Normalize feature axis."""
        if axis is None or original_rank is None or feature_rank is None:
            return None

        normalized = axis
        if normalized < 0:
            normalized = original_rank + normalized

        if normalized < 0 or normalized >= original_rank:
            return None

        if original_rank == feature_rank + 1:
            if normalized == 0:
                return None
            return normalized - 1
        if normalized >= feature_rank:
            return None
        return normalized

    def resolve_relaxed_feature_axis(self, shape, merge_context):
        """Resolve relaxed feature axis."""
        if shape is None:
            return None

        feature_shape = self.get_feature_shape(shape)
        if feature_shape is None:
            return None

        concat_axis = None
        if merge_context is not None:
            concat_axis = merge_context.get("concat_axis")

        return self.normalize_feature_axis(
            axis=concat_axis,
            original_rank=len(shape),
            feature_rank=len(feature_shape),
        )
