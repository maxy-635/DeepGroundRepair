from step_two.merge_shape_debug.locator.target_api_locator_utils import (
    get_kernel_pair,
    get_pool_size_pair,
    get_stride_pair,
    is_transpose_conv,
    normalize_padding,
    parse_call_config,
)


class SpatialPreservingConvHeuristic:
    """Implement the spatial preserving conv heuristic component."""

    def __init__(self, rule_registry):
        """Initialize the instance."""
        self.rule_registry = rule_registry

    def resolve_layer_rule(self, layer_type):
        """Resolve layer rule."""
        if not layer_type:
            return None

        semantic_type = self.rule_registry["layer_type_map"].get(layer_type)
        if semantic_type is None:
            return None
        return {"semantic_type": semantic_type}

    def is_conv_api(self, api_name):
        rule_info = self.resolve_layer_rule(api_name)
        return bool(rule_info) and rule_info.get("semantic_type") == "conv"

    def is_strong_shape_preserving_conv(self, call_info):
        api_name = call_info.get("api_name")
        if not self.is_conv_api(api_name) or is_transpose_conv(api_name):
            return False

        config = parse_call_config(call_info.get("original_call"), api_name)
        stride_pair = get_stride_pair(config, api_name=api_name)
        if stride_pair != (1, 1):
            return False

        kernel_pair = get_kernel_pair(config)
        padding = normalize_padding(config.get("padding"))

        if kernel_pair == (1, 1):
            return True
        if padding == "same":
            return True
        if kernel_pair == (3, 3) and padding == (1, 1):
            return True
        return False

    def is_pool_spatial_changer(self, call_info):
        api_name = call_info.get("api_name") or ""
        lowered = api_name.lower()
        if "global" in lowered or "adaptive" in lowered:
            return True

        config = parse_call_config(call_info.get("original_call"), api_name)
        pool_size_pair = get_pool_size_pair(config)
        stride_pair = get_stride_pair(config, api_name=api_name)
        padding = normalize_padding(config.get("padding"))

        if stride_pair is not None and any(item > 1 for item in stride_pair):
            return True
        if pool_size_pair is not None and any(item > 1 for item in pool_size_pair):
            if padding == "valid" or padding is None:
                return True
        return False

    def is_conv_spatial_changer(self, call_info):
        api_name = call_info.get("api_name")
        if is_transpose_conv(api_name):
            return True

        config = parse_call_config(call_info.get("original_call"), api_name)
        stride_pair = get_stride_pair(config, api_name=api_name)
        kernel_pair = get_kernel_pair(config)
        padding = normalize_padding(config.get("padding"))

        if stride_pair is not None and any(item > 1 for item in stride_pair):
            return True
        if kernel_pair is not None and any(item > 1 for item in kernel_pair):
            if padding == "valid":
                return True
        return False

    def is_stronger_spatial_changer(self, call_info):
        api_name = call_info.get("api_name")
        rule_info = self.resolve_layer_rule(api_name)
        if rule_info is None:
            return False

        semantic_type = rule_info.get("semantic_type")
        if semantic_type == "pool":
            return self.is_pool_spatial_changer(call_info)
        if semantic_type == "upsample":
            return True
        if semantic_type == "conv":
            return self.is_conv_spatial_changer(call_info)
        return False

    def should_skip_preserving_conv(self, effective_calls, current_index, call_info, mismatch_type):
        if mismatch_type != "spatial":
            return False
        if not self.is_conv_api(call_info.get("api_name")):
            return False
        if not self.is_strong_shape_preserving_conv(call_info):
            return False

        earlier_calls = effective_calls[current_index + 1 :]
        return any(self.is_stronger_spatial_changer(item) for item in earlier_calls)
