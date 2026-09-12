from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import Any


API_NAME_ALIASES = {
    "keras.layers.Conv2D": "tf.keras.layers.Conv2D",
    "tensorflow.keras.layers.Conv2D": "tf.keras.layers.Conv2D",
    "tf.keras.layers.Conv2D": "tf.keras.layers.Conv2D",
    "tf.nn.conv2d": "tf.nn.conv2d",
    "torch.nn.Conv2d": "torch.nn.Conv2d",
    "paddle.nn.Conv2D": "paddle.nn.Conv2D",
    "keras.layers.MaxPooling2D": "tf.keras.layers.MaxPooling2D",
    "keras.layers.MaxPool2D": "tf.keras.layers.MaxPooling2D",
    "tensorflow.keras.layers.MaxPooling2D": "tf.keras.layers.MaxPooling2D",
    "tensorflow.keras.layers.MaxPool2D": "tf.keras.layers.MaxPooling2D",
    "tf.keras.layers.MaxPooling2D": "tf.keras.layers.MaxPooling2D",
    "tf.keras.layers.MaxPool2D": "tf.keras.layers.MaxPooling2D",
    "tf.nn.max_pool2d": "tf.nn.max_pool2d",
    "torch.nn.MaxPool2d": "torch.nn.MaxPool2d",
    "paddle.nn.MaxPool2D": "paddle.nn.MaxPool2D",
    "keras.layers.AveragePooling2D": "tf.keras.layers.AveragePooling2D",
    "keras.layers.AvgPool2D": "tf.keras.layers.AveragePooling2D",
    "tensorflow.keras.layers.AveragePooling2D": "tf.keras.layers.AveragePooling2D",
    "tensorflow.keras.layers.AvgPool2D": "tf.keras.layers.AveragePooling2D",
    "tf.keras.layers.AveragePooling2D": "tf.keras.layers.AveragePooling2D",
    "tf.keras.layers.AvgPool2D": "tf.keras.layers.AveragePooling2D",
    "tf.nn.avg_pool2d": "tf.nn.avg_pool2d",
    "torch.nn.AvgPool2d": "torch.nn.AvgPool2d",
    "paddle.nn.AvgPool2D": "paddle.nn.AvgPool2D",
}


POSITIONAL_PARAM_NAMES = {
    "torch.nn.Conv2d": ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
    "paddle.nn.Conv2D": ["in_channels", "out_channels", "kernel_size", "stride", "padding"],
    "tf.keras.layers.Conv2D": ["filters", "kernel_size", "strides", "padding"],
    "torch.nn.MaxPool2d": ["kernel_size", "stride", "padding"],
    "torch.nn.AvgPool2d": ["kernel_size", "stride", "padding"],
    "paddle.nn.MaxPool2D": ["kernel_size", "stride", "padding"],
    "paddle.nn.AvgPool2D": ["kernel_size", "stride", "padding"],
    "tf.keras.layers.MaxPooling2D": ["pool_size", "strides", "padding"],
    "tf.keras.layers.AveragePooling2D": ["pool_size", "strides", "padding"],
}


def find_shape_pair(
    shape_pairs: list[dict[str, Any]],
    buggy_api_call: dict[str, Any],
) -> dict[str, Any] | None:
    """Find shape pair."""
    buggy_api_call_index = buggy_api_call.get("_buggy_api_call_index")
    if not isinstance(buggy_api_call_index, int):
        return None

    for shape_pair in shape_pairs:
        if shape_pair.get("buggy_api_call_index") == buggy_api_call_index:
            return shape_pair
    return None


def resolve_api_name(
    buggy_api_call: dict[str, Any],
    source_code: str,
    supported_api_names: set[str],
) -> str | None:
    """Resolve api name."""
    call_api_name = read_api_name_from_call(
        buggy_api_call.get("normalized_call") or buggy_api_call.get("original_call"),
        source_code,
    )

    for api_name in (
        call_api_name,
        buggy_api_call.get("api_name"),
        buggy_api_call.get("_api_name"),
    ):
        if not api_name:
            continue

        normalized_api_name = API_NAME_ALIASES.get(api_name, api_name)
        if normalized_api_name in supported_api_names:
            return normalized_api_name

    return None


def extract_call_params(buggy_api_call: dict[str, Any], api_name: str) -> dict[str, Any]:
    """Extract call params."""
    call_node = parse_call_node(
        buggy_api_call.get("normalized_call") or buggy_api_call.get("original_call")
    )
    if call_node is None:
        return {}

    params = {}
    positional_names = POSITIONAL_PARAM_NAMES.get(api_name, [])
    for index, arg in enumerate(call_node.args):
        if index < len(positional_names):
            params[positional_names[index]] = safe_literal_eval(arg)

    for keyword in call_node.keywords:
        if keyword.arg is not None:
            params[keyword.arg] = safe_literal_eval(keyword.value)

    return {name: value for name, value in params.items() if value is not None}


def read_api_name_from_call(call_source: str | None, source_code: str) -> str | None:
    """Read api name from call."""
    call_node = parse_call_node(call_source)
    if call_node is None:
        return None

    raw_name = ast.unparse(call_node.func)
    parts = raw_name.split(".")
    import_aliases = collect_import_aliases(source_code)
    if parts and parts[0] in import_aliases:
        return ".".join(import_aliases[parts[0]].split(".") + parts[1:])
    return raw_name


def parse_call_node(call_source: str | None) -> ast.Call | None:
    """Parse call node."""
    if not call_source:
        return None

    try:
        tree = ast.parse(call_source)
    except SyntaxError:
        return None
    if not tree.body:
        return None

    stmt = tree.body[0]
    value = getattr(stmt, "value", None)
    if isinstance(stmt, ast.Assign):
        value = stmt.value
    if not isinstance(value, ast.Call):
        return None

    return value.func if isinstance(value.func, ast.Call) else value


def collect_import_aliases(source_code: str) -> dict[str, str]:
    """Collect import aliases."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return {}

    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                local_name = item.asname or item.name.split(".")[0]
                aliases[local_name] = item.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for item in node.names:
                local_name = item.asname or item.name
                aliases[local_name] = f"{node.module}.{item.name}"
    return aliases


def safe_literal_eval(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def resolve_merge_ignored_axes(
    merge_context: dict[str, Any],
    rank: int,
) -> set[int]:
    """Resolve merge ignored axes."""
    merge_kind = merge_context.get("merge_kind")
    concat_axis = merge_context.get("concat_axis")
    if merge_kind != "Concatenate" or not isinstance(concat_axis, int):
        return set()

    axis = concat_axis if concat_axis >= 0 else rank + concat_axis
    if 0 <= axis < rank:
        return {axis}
    return set()


def assert_same_shape(
    actual: Sequence[Any],
    expected: Sequence[Any],
    ignored_axes: set[int] | None = None,
) -> bool:
    """Assert same shape."""
    if len(actual) != len(expected):
        return False

    ignored_axes = ignored_axes or set()
    for index, (actual_dim, expected_dim) in enumerate(zip(actual, expected)):
        if index in ignored_axes:
            continue
        if actual_dim is None or expected_dim is None:
            continue
        if int(actual_dim) != int(expected_dim):
            return False
    return True
