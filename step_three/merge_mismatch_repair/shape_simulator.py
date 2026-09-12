from __future__ import annotations
import math
from collections.abc import Sequence
from typing import Any

Pair = tuple[int, int]


class ShapeSimulator:
    """Implement the shape simulator component."""

    API_TO_OP = {
        "torch.nn.Conv2d": "conv2d",
        "tf.keras.layers.Conv2D": "conv2d",
        "tf.nn.conv2d": "conv2d",
        "paddle.nn.Conv2D": "conv2d",
        "torch.nn.MaxPool2d": "pool2d",
        "tf.keras.layers.MaxPooling2D": "pool2d",
        "tf.nn.max_pool2d": "pool2d",
        "paddle.nn.MaxPool2D": "pool2d",
        "torch.nn.AvgPool2d": "pool2d",
        "tf.keras.layers.AveragePooling2D": "pool2d",
        "tf.nn.avg_pool2d": "pool2d",
        "paddle.nn.AvgPool2D": "pool2d",
    }

    def simulate(self, api_name: str, input_shape: Sequence[int], **params: Any) -> list[int]:

        op = self.API_TO_OP.get(api_name)
        if op is None:
            raise KeyError(f"No shape simulator registered for {api_name!r}")

        data_format = "channels_last" if api_name.startswith("tf.") else "channels_first"
        if op == "conv2d":
            return self.conv2d(input_shape, data_format=data_format, **params)
        if op == "pool2d":
            return self.pool2d(input_shape, data_format=data_format, **params)

    def conv2d(
        self,
        input_shape: Sequence[int],
        out_channels: int | None = None,
        kernel_size: int | Sequence[int] | None = None,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] | str = 0,
        filters: int | Sequence[int] | None = None,
        strides: int | Sequence[int] | None = None,
        data_format: str = "channels_first",
        **_: Any,
    ) -> list[int]:
        batch, _, height, width = self._split_shape(input_shape, data_format)
        kernel = self._kernel_size(kernel_size, filters)
        stride = self._pair(strides or stride)
        out_channels = self._out_channels(out_channels, filters)
        output_hw = self._forward_output((height, width), kernel, stride, padding)
        return self._shape(batch, out_channels, output_hw, data_format)

    def pool2d(
        self,
        input_shape: Sequence[int],
        kernel_size: int | Sequence[int] | None = None,
        stride: int | Sequence[int] | None = None,
        padding: int | Sequence[int] | str = 0,
        ksize: int | Sequence[int] | None = None,
        pool_size: int | Sequence[int] | None = None,
        strides: int | Sequence[int] | None = None,
        data_format: str = "channels_first",
        **_: Any,
    ) -> list[int]:
        batch, channels, height, width = self._split_shape(input_shape, data_format)
        kernel = self._pair(kernel_size or ksize or pool_size)
        stride = self._pair(strides or stride or kernel)
        output_hw = self._forward_output((height, width), kernel, stride, padding)
        return self._shape(batch, channels, output_hw, data_format)

    @staticmethod
    def _split_shape(input_shape: Sequence[int], data_format: str) -> tuple[int, int, int, int]:
        if data_format == "channels_last":
            batch, height, width, channels = input_shape
            return int(batch), int(channels), int(height), int(width)

        batch, channels, height, width = input_shape
        return int(batch), int(channels), int(height), int(width)

    @staticmethod
    def _shape(batch: int, channels: int, hw: Pair, data_format: str) -> list[int]:
        height, width = hw
        if data_format == "channels_last":
            return [batch, height, width, channels]
        return [batch, channels, height, width]

    @staticmethod
    def _pair(value: int | Sequence[int] | None) -> Pair:
        if isinstance(value, int):
            return value, value
        if value is None:
            raise ValueError("The parameter must be an integer or a sequence of length 2")
        return int(value[0]), int(value[1])

    @staticmethod
    def _out_channels(out_channels: int | None, filters: int | Sequence[int] | None) -> int:
        if out_channels is not None:
            return int(out_channels)
        if isinstance(filters, int):
            return filters
        if filters is not None:
            return int(filters[-1])
        raise ValueError("A convolution layer must provide out_channels or filters")

    def _kernel_size(
        self,
        kernel_size: int | Sequence[int] | None,
        filters: int | Sequence[int] | None,
    ) -> Pair:
        if kernel_size is not None:
            return self._pair(kernel_size)
        if isinstance(filters, Sequence) and not isinstance(filters, (str, bytes)):
            return self._pair(filters[:2])
        raise ValueError("A convolution layer must provide kernel_size or a filter shape")

    def _forward_output(
        self,
        input_hw: Pair,
        kernel: Pair,
        stride: Pair,
        padding: int | Sequence[int] | str,
    ) -> Pair:
        if isinstance(padding, str):
            padding = padding.lower()
            if padding == "same":
                return math.ceil(input_hw[0] / stride[0]), math.ceil(input_hw[1] / stride[1])
            if padding == "valid":
                padding = (0, 0)
            else:
                raise ValueError(f"Unsupported padding string: {padding!r}")

        # output = floor((input + 2*padding - kernel) / stride + 1)
        pad_h, pad_w = self._pair(padding)
        return (
            math.floor((input_hw[0] + 2 * pad_h - kernel[0]) / stride[0] + 1),
            math.floor((input_hw[1] + 2 * pad_w - kernel[1]) / stride[1] + 1),
        )
