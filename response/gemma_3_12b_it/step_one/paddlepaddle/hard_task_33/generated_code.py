import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class Block(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.depthwise = nn.DepthwiseConv2D(kernel_size=3, stride=1, padding=1, groups=1)
        self.conv2 = nn.Conv2D(out_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.depthwise(x)
        x = self.conv2(x)
        x = x + identity
        return x


class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.branch1 = Block(in_channels=1, out_channels=32)
        self.branch2 = Block(in_channels=1, out_channels=32)
        self.branch3 = Block(in_channels=1, out_channels=32)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=32 * 7 * 7, out_features=10)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        merged = paddle.concat([out1, out2, out3], axis=1)
        flattened = self.flatten(merged)
        output = self.linear(flattened)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)