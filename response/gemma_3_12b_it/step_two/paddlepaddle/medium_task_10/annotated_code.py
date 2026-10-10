import paddle
import paddle.nn as nn

class BasicBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2D(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        self.shortcut = nn.Conv2D(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn_shortcut = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        # the initial shape of x is: (1, 16, 32, 32)
        identity = self.shortcut(x)  # output tensor shape: (1, 16, 32, 32)
        identity = self.bn_shortcut(identity)  # output tensor shape: (1, 16, 32, 32)
        out = self.conv1(x)  # output tensor shape: (1, 16, 32, 32)
        out = self.bn1(out)  # output tensor shape: (1, 16, 32, 32)
        out = self.relu(out)  # output tensor shape: (1, 16, 32, 32)
        out = paddle.add(out, identity)  # output tensor shape: (1, 16, 32, 32)
        return out


class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2D(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(16)
        self.relu = nn.ReLU()

        self.level1 = BasicBlock(16, 16)

        self.level2_conv1 = nn.Conv2D(16, 32, kernel_size=3, stride=1, padding=1)
        self.level2_bn1 = nn.BatchNorm2D(32)
        self.level2_conv2 = nn.Conv2D(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')
        self.level2_bn2 = nn.BatchNorm2D(32)
        self.level2_block1 = BasicBlock(32, 32)
        self.level2_block2 = BasicBlock(32, 32)

        self.level3_conv1 = nn.Conv2D(16, 64, kernel_size=3, stride=1, padding=1)
        self.level3_bn1 = nn.BatchNorm2D(64)

        self.avg_pool = nn.AvgPool2D(kernel_size=8, stride=1, padding=0)
        self.linear = nn.Linear(64, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x = self.conv1(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.bn1(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.relu(x)  # output tensor shape: (1, 16, 32, 32)

        x = self.level1(x)  # output tensor shape: (1, 16, 32, 32)

        x = self.level2_conv1(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.level2_bn1(x)  # output tensor shape: (1, 32, 32, 32)
        identity = x  # output tensor shape: (1, 32, 32, 32)
        x = self.level2_conv2(x)
        x = self.level2_bn2(x)
        x = self.level2_block1(x)
        x = paddle.add(identity, x)
        identity = x
        x = self.level2_block2(x)
        x = paddle.add(identity, x)

        x = self.level3_conv1(x)
        x = self.level3_bn1(x)
        x = paddle.add(x, self.level1(x))

        x = self.avg_pool(x)
        x = paddle.flatten(x, start=1)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)