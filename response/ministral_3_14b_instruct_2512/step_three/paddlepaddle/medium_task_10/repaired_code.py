import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class BasicBlock(nn.Layer):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )
        self.bn1 = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2D(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2D(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0
                ),
                nn.BatchNorm2D(out_channels)
            )

    def forward(self, x):
        # the initial shape of x is: (1, 16, 32, 32)
        out = self.conv1(x)  # output tensor shape: (1, 32, 16, 16)
        out = self.bn1(out)  # output tensor shape: (1, 32, 16, 16)
        out = self.relu(out)  # output tensor shape: (1, 32, 16, 16)
        out = self.conv2(out)  # output tensor shape: (1, 32, 16, 16)
        out = self.bn2(out)  # output tensor shape: (1, 32, 16, 16)
        shortcut = self.shortcut(x)  # output tensor shape: (1, 32, 16, 16)
        out += shortcut  # output tensor shape: (1, 32, 16, 16)
        out = self.relu(out)  # output tensor shape: (1, 32, 16, 16)
        return out

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolution layer
        self.conv1 = nn.Conv2D(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn1 = nn.BatchNorm2D(16)
        self.relu = nn.ReLU()

        # First level residual block
        self.block1 = BasicBlock(in_channels=16, out_channels=16)

        # Second level residual blocks
        self.block2_1 = BasicBlock(in_channels=16, out_channels=32, stride=2)
        self.block2_2 = BasicBlock(in_channels=32, out_channels=32)

        # Third level global branch
        self.global_conv = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)






        self.global_bn = nn.BatchNorm2D(32)

        # Final layers
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.fc = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Initial processing
        x = self.conv1(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.bn1(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.relu(x)  # output tensor shape: (1, 16, 32, 32)

        # First level residual connection
        x = self.block1(x)  # output tensor shape: (1, 16, 32, 32)

        # Second level residual connection
        x = self.block2_1(x)  # output tensor shape: (1, 32, 16, 16)
        x = self.block2_2(x)  # output tensor shape: (1, 32, 16, 16)

        # Third level global branch
        global_branch = self.global_conv(x)
        global_branch = self.global_bn(global_branch)

        # Combine with second level output
        x = x + global_branch

        # Final processing
        x = self.avg_pool(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))  # CIFAR-10 image shape
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # Should be [1, 10]