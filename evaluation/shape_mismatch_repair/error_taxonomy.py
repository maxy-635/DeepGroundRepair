import re
from typing import Any


SHAPE_MISMATCH_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "pytorch_matmul_shape",
        re.compile(r"shapes cannot be multiplied", re.IGNORECASE),
    ),
    (
        "pytorch_conv1d_rank_mismatch",
        re.compile(
            r"expected 2d \(unbatched\) or 3d \(batched\) input to conv1d",
            re.IGNORECASE,
        ),
    ),
    (
        "pytorch_conv_channel_mismatch",
        re.compile(
            r"expected input.*to have \d+ channels, but got \d+ channels instead",
            re.IGNORECASE,
        ),
    ),
    (
        "keras_inputs_incompatible_shapes",
        re.compile(r"inputs have incompatible shapes", re.IGNORECASE),
    ),
    (
        "tensorflow_incompatible_shapes",
        re.compile(
            r"incompatible shapes:\s*\[[^\]]+\]\s*(?:vs\.?|and)\s*\[[^\]]+\]",
            re.IGNORECASE,
        ),
    ),
    (
        "paddle_linear_dim_mismatch",
        re.compile(
            r"input\(y\) has error dim\. y'dims\[0\] must be equal to",
            re.IGNORECASE,
        ),
    ),
    (
        "layer_input_incompatible",
        re.compile(
            r'input 0 of layer .* is incompatible with the layer',
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "concat_matching_shapes",
        re.compile(r"requires inputs with matching shapes", re.IGNORECASE),
    ),
    (
        "tensorflow_concat_dimension_mismatch",
        re.compile(
            r"concatop\s*:\s*dimension \d+ .*must be equal",
            re.IGNORECASE,
        ),
    ),
    (
        "tensor_size_mismatch",
        re.compile(r"must match the size of tensor", re.IGNORECASE),
    ),

    (
        "paddle_broadcast_dim_mismatch",
        re.compile(
            r"broadcast dimension mismatch",
            re.IGNORECASE,
        ),
    ),
    (
        "paddle_operands_broadcast_failure",
        re.compile(
            r"operands could not be broadcast together",
            re.IGNORECASE,
        ),
    ),
    (
        "paddle_concat_expected_equal_dim",
        re.compile(
            r"dimension of input\[\d+\] and input\[\d+\] is expected to be equal",
            re.IGNORECASE,
        ),
    ),
    (
        "paddle_concat_inputs_dims_mismatch",
        re.compile(
            r"expected inputs_dims\[0\]\[j\] == inputs_dims\[i\]\[j\]",
            re.IGNORECASE,
        ),
    ),
    (
        "paddle_channel_mismatch",
        re.compile(
            r"input's channels should be equal to filter's channels",
            re.IGNORECASE,
        ),
    ),
    (
        # input_channels != filter_channels * groups.
        "paddle_channel_hint",
        re.compile(
            r"expected input_channels == filter_channels \* groups",
            re.IGNORECASE,
        ),
    ),
    (
        "tensor_size_must_match",
        re.compile(r"(sizes|size) of tensors? must match", re.IGNORECASE),
    ),

    (
        "pytorch_pool_output_too_small",
        re.compile(
            r"calculated output size: .*output size is too small",
            re.IGNORECASE,
        ),
    ),
    (
        "pytorch_batchnorm_channel_mismatch",
        re.compile(
            r"running_mean should contain \d+ elements not \d+",
            re.IGNORECASE,
        ),
    ),
    (
        "pytorch_conv2d_rank_mismatch",
        re.compile(
            r"expected 3d \(unbatched\) or 4d \(batched\) input to conv2d",
            re.IGNORECASE,
        ),
    ),
    (
        "paddle_reshape_size_mismatch",
        re.compile(
            r"output_shape\[unk_dim_idx\] \* capacity.*!= in_size",
            re.IGNORECASE,
        ),
    ),
    (
        "tensorflow_reshape_value_count_mismatch",
        re.compile(
            r"input to reshape is a tensor with \d+ values, but the requested shape",
            re.IGNORECASE,
        ),
    ),
    (
        "paddle_reshape_capacity_mismatch",
        re.compile(
            r"input tensor x'?s?size must be equal to the capacity of 'shape'",
            re.IGNORECASE,
        ),
    ),
    (
        "pytorch_invalid_reshape_size",
        re.compile(
            r"shape '.*' is invalid for input of size \d+",
            re.IGNORECASE,
        ),
    ),
    (
        "pytorch_layernorm_normalized_shape_mismatch",
        re.compile(
            r"given normalized_shape=.*expected input with shape .*but got input of size",
            re.IGNORECASE,
        ),
    ),
    (
        "incompatible_input_shape",
        re.compile(
            r"expected axis .* input shape .* but received input with shape",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "dimensions_must_be_equal",
        re.compile(r"dimensions must be equal", re.IGNORECASE),
    ),
    (
        "tensorflow_downsampling_output_nonpositive",
        re.compile(
            r"dimensions in the output is <= 0 .*received input shape .*output shape",
            re.IGNORECASE,
        ),
    ),
    (
        "pytorch_dimension_out_of_range",
        re.compile(
            r"dimension out of range \(expected to be in range of \[-?\d+, \d+\], but got \d+\)",
            re.IGNORECASE,
        ),
    ),
    (
        "paddle_axis_or_split_shape_mismatch",
        re.compile(
            r"(start_axis should be .*range \[-rank\(x\), rank\(x\)\)|"
            r"input's size along the split dimension must be evenly divisible)",
            re.IGNORECASE,
        ),
    ),
    (
        "expected_size_got_size",
        re.compile(r"expected size .* got size", re.IGNORECASE),
    ),
]


def normalize_error_text(error_text: Any) -> str:
    """Normalize error text."""
    if error_text is None:
        return ""
    if isinstance(error_text, str):
        if error_text.strip().lower() == "none":
            return ""
        return error_text
    return str(error_text)


def classify_shape_mismatch(error_text: Any) -> tuple[bool, str | None]:
    """Classify shape mismatch."""
    text = normalize_error_text(error_text)
    if not text:
        return False, None

    for rule_name, pattern in SHAPE_MISMATCH_RULES:
        if pattern.search(text):
            return True, rule_name
    return False, None
