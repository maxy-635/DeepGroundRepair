from collections import Counter
from step_two.merge_shape_debug.analysis.tensor_shape_normalizer import TensorShapeNormalizer


class ShapeCompare:
    """Implement the shape compare component."""

    def __init__(self):
        """Initialize the instance."""
        self.normalizer = TensorShapeNormalizer()

    def has_spatial_mismatch(self, branches, merge_context):
        comparable_dims = []
        for branch in branches:
            feature_axes = self.normalizer.describe_feature_axes(branch["shape"], merge_context)
            if feature_axes is None or not feature_axes["spatial_indexes"]:
                continue
            comparable_dims.append(feature_axes)

        if len(comparable_dims) < 2:
            return False

        axis = self.resolve_relaxed_feature_axis(branches, merge_context)

        reference_rank = len(comparable_dims[0]["feature_shape"])
        comparable_dims = [
            shape_info
            for shape_info in comparable_dims
            if len(shape_info["feature_shape"]) == reference_rank
        ]
        if len(comparable_dims) < 2:
            return False

        spatial_indexes = comparable_dims[0]["spatial_indexes"]
        for dim_index in spatial_indexes:
            if not self.requires_equal_dim(merge_context["merge_kind"], axis, dim_index):
                continue
            values = [shape_info["feature_shape"][dim_index] for shape_info in comparable_dims]
            if len(set(values)) > 1:
                return True
        return False

    def has_channel_mismatch(self, branches, merge_context):
        comparable_dims = []
        for branch in branches:
            feature_axes = self.normalizer.describe_feature_axes(branch["shape"], merge_context)
            if feature_axes is None or feature_axes["channel_index"] is None:
                continue
            comparable_dims.append(feature_axes)

        if len(comparable_dims) < 2:
            return False

        axis = self.resolve_relaxed_feature_axis(branches, merge_context)
        reference_rank = len(comparable_dims[0]["feature_shape"])
        comparable_dims = [
            shape_info
            for shape_info in comparable_dims
            if len(shape_info["feature_shape"]) == reference_rank
        ]
        if len(comparable_dims) < 2:
            return False

        channel_index = comparable_dims[0]["channel_index"]
        if not self.requires_equal_dim(merge_context["merge_kind"], axis, channel_index):
            return False

        values = [shape_info["feature_shape"][channel_index] for shape_info in comparable_dims]
        return len(set(values)) > 1

    def resolve_relaxed_feature_axis(self, branches, merge_context):
        """Resolve relaxed feature axis."""
        reference_rank = self.get_shape_rank(branches)
        if reference_rank is None:
            return None

        for branch in branches:
            shape = branch.get("shape")
            if shape is None or len(shape) != reference_rank:
                continue
            return self.normalizer.resolve_relaxed_feature_axis(shape, merge_context)

        return None

    def build_majority_profile(self, branches, mismatch_types, merge_context=None):
        """Build majority profile."""
        profile = {}
        threshold = len(branches) / 2.0

        if "spatial" in mismatch_types:
            spatial_signatures = []
            for branch in branches:
                signature = self.get_spatial_signature(branch["shape"], merge_context)
                if signature is not None:
                    spatial_signatures.append(signature)
            mode, count = self.mode_with_count(spatial_signatures)
            if count > threshold:
                profile["spatial"] = mode

        if "channel" in mismatch_types:
            channel_signatures = []
            for branch in branches:
                signature = self.get_channel_signature(branch["shape"], merge_context)
                if signature is not None:
                    channel_signatures.append(signature)
            mode, count = self.mode_with_count(channel_signatures)
            if count > threshold:
                profile["channel"] = mode

        return profile

    def requires_equal_dim(self, merge_kind, axis, dim_index):
        kind = str(merge_kind).lower()
        if kind in {"cat", "concatenate", "concat"} and axis == dim_index:
            return False
        return True

    def get_shape_rank(self, branches):
        """Return shape rank."""
        ranks = []
        for branch in branches:
            if branch["shape"] is None:
                continue
            ranks.append(len(branch["shape"]))
        if not ranks:
            return None
        return Counter(ranks).most_common(1)[0][0]

    def get_spatial_signature(self, shape, merge_context=None):
        """Return spatial signature."""
        feature_axes = self.normalizer.describe_feature_axes(shape, merge_context)
        if feature_axes is None or not feature_axes["spatial_indexes"]:
            return None
        return tuple(feature_axes["feature_shape"][index] for index in feature_axes["spatial_indexes"])

    def get_channel_signature(self, shape, merge_context=None):
        """Return channel signature."""
        feature_axes = self.normalizer.describe_feature_axes(shape, merge_context)
        if feature_axes is None or feature_axes["channel_index"] is None:
            return None
        return feature_axes["feature_shape"][feature_axes["channel_index"]]

    def mode_with_count(self, values):
        if not values:
            return None, 0
        counter = Counter(values)
        value, count = counter.most_common(1)[0]
        return value, count

    def compute_deviation_score(self, shape, majority_profile, merge_context=None):
        """Compute deviation score."""
        score = 0

        if "spatial" in majority_profile:
            if self.get_spatial_signature(shape, merge_context) != majority_profile["spatial"]:
                score += 1

        if "channel" in majority_profile:
            if self.get_channel_signature(shape, merge_context) != majority_profile["channel"]:
                score += 1

        return score
