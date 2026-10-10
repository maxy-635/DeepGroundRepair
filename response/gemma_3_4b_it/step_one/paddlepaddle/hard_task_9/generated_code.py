import sys
import paddle
from paddle.nn import Layer
from paddle.nn import Conv2D
from paddle.nn import Flatten
from paddle import concat
from paddle import add
from paddle import dist
from paddle import identity

class DL_Model(Layer):
    def __init__(self):
        super().__init__()
        self.branch1 = Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch2 = Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch3 = Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.concat = concat([1])
        self.conv_adjust = Conv2D(in_channels=192, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.fc1 = LayerNorm(3)
        self.fc2 = Conv2D(in_channels=3, out_channels=10, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        merged = add(x, branch1)
        merged = add(merged, branch2)
        merged = add(merged, branch3)

        merged = self.concat(merged)
        merged = self.conv_adjust(merged)

        output = self.fc1(merged)
        output = self.fc2(output)
        return output

    def __repr__(self):
        return type(self).__name__ + "(" + ", ".join([str(arg) for arg in self.parameters()]) + ")"

class LayerNorm(Layer):
    def __init__(self, normalized_shape):
        super().__init__()
        self.gamma = paddle.nn.Parameter(paddle.ones(normalized_shape))
        self.beta = paddle.nn.Parameter(paddle.zeros(normalized_shape))

    def forward(self, x):
        mean = paddle.mean(x, axis=-1, keepdim=True)
        variance = paddle.var(x, axis=-1, keepdim=True)
        return self.gamma * (x - mean) / paddle.sqrt(variance + 1e-5) + self.beta

if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for validation
    input_tensor = paddle.randn(batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)