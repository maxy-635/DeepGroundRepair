import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class BasicBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn1 = nn.BatchNorm2D(num_features=out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BayerNorm2D(num_features='mask')

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.initial_conv = nn.Conv2D(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn_initial = nn.BatchNorm2D(num_features=16)
        self.relu_initial = nn.ReLU()

        self.block1 = BasicBlock(in_channels=16, out_channels=16)
        self.block2 = BasicBlock(in_channels=16, out_channels=16)

        self.branch_conv = nn.Conv2D(
            in_channels=16,
            out_channels=16,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=16, out_features=10)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.bn_initial(x)
        x = self.relu_initial(x)

        x = self.block1(x)
        x = self.block2(x)

        branch = self.branch_conv(x)
        x += branch

        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # CIFAR-10 image shape: (batch_size, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)