import ast


def expand_container_calls(producer_calls, init_definitions):
    expanded = []
    for call_info in producer_calls:
        init_definition = init_definitions.get(call_info["layer_name"])
        if init_definition is None or not getattr(init_definition, "children", None):
            expanded.append(call_info)
            continue

        for child in reversed(init_definition.children):
            expanded.append(
                {
                    "layer_name": child["layer_name"],
                    "api_name": child["api_name"],
                    "call_line_no": call_info["call_line_no"],
                    "init_line_no": child["init_line_no"],
                    "end_line_no": child.get("end_line_no"),
                    "original_call": child["original_call"],
                    "container_name": child["container_name"],
                    "container_init_line_no": child["container_init_line_no"],
                    "child_index": child["child_index"],
                }
            )
    return expanded


def merge_call_line_no(existing_line_no, new_line_no):
    """Merge call line no."""
    if isinstance(existing_line_no, int) and isinstance(new_line_no, int):
        return max(existing_line_no, new_line_no)
    if isinstance(existing_line_no, int):
        return existing_line_no
    if isinstance(new_line_no, int):
        return new_line_no
    return None


def strip_internal_fields(record):
    """Strip internal fields."""
    return {
        key: value
        for key, value in record.items()
        if not key.startswith("_")
    }


def collect_params_to_mask(target_records):
    """Collect params to mask."""
    params = []
    for target_record in target_records:
        params.extend(target_record.get("params_to_mask") or [])
    return unique_list(params)


def get_constructor_call_node(original_call, api_name):
    """Return constructor call node."""
    if not original_call or not api_name:
        return None

    try:
        tree = ast.parse(original_call)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if get_called_api_name(node) == api_name:
            return node
    return None


def get_called_api_name(call_node):
    """Return called api name."""
    func = getattr(call_node, "func", None)
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def parse_call_config(original_call, api_name):
    """Parse call config."""
    call_node = get_constructor_call_node(original_call, api_name)
    if call_node is None:
        return {}

    config = {}
    positional_names = get_common_positional_parameter_names(api_name)
    for index, arg in enumerate(call_node.args):
        if index >= len(positional_names):
            break
        value = literal_value_or_none(arg)
        if value is not None:
            config[positional_names[index]] = value

    for keyword in call_node.keywords:
        if keyword.arg is None:
            continue
        value = literal_value_or_none(keyword.value)
        if value is not None:
            config[keyword.arg] = value

    return config


def get_common_positional_parameter_names(api_name):
    """Return common positional parameter names."""
    if not api_name:
        return []

    lowered = api_name.lower()
    if "pool" in lowered:
        return ["pool_size", "stride", "padding"]
    if "convtranspose" in lowered or "transpose" in lowered:
        return ["filters_or_out_channels", "kernel_size", "stride", "padding"]
    if "conv" in lowered:
        return ["filters_or_out_channels", "kernel_size", "stride", "padding"]
    return []


def literal_value_or_none(value_node):
    try:
        return ast.literal_eval(value_node)
    except Exception:
        return None


def normalize_scalar_or_pair(value):
    """Normalize scalar or pair."""
    if isinstance(value, (int, float)):
        normalized = int(value)
        return (normalized, normalized)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        if all(isinstance(item, (int, float)) for item in value):
            return (int(value[0]), int(value[1]))
    return None


def get_stride_pair(config, api_name=None):
    """Return stride pair."""
    stride = config.get("stride")
    if stride is None:
        stride = config.get("strides")
    stride_pair = normalize_scalar_or_pair(stride)
    if stride_pair is not None:
        return stride_pair

    lowered = (api_name or "").lower()
    if "conv" in lowered and "transpose" not in lowered:
        return (1, 1)
    if "pool" in lowered:
        return get_pool_size_pair(config)
    return None


def get_kernel_pair(config):
    """Return kernel pair."""
    return normalize_scalar_or_pair(config.get("kernel_size"))


def get_pool_size_pair(config):
    """Return pool size pair."""
    return normalize_scalar_or_pair(config.get("pool_size"))


def normalize_padding(padding):
    """Normalize padding."""
    if isinstance(padding, str):
        return padding.lower()
    return normalize_scalar_or_pair(padding)


def is_transpose_conv(api_name):
    lowered = (api_name or "").lower()
    return "transpose" in lowered

def unique_list(values):
    result = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
