import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class Block(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv1x1_1 = nn.Conv2D(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
        self.depthwise_conv = nn.Conv2D(in_channels=out_channels,
                                       out_channels=out_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       groups=out_channels)
        self.conv1x1_2 = nn.Conv2D(in_channels=out_channels,
                                  out_channels=in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0)
        self.bn1 = nn.BatchNorm2D(out_channels)
        self.bn2 = nn.BatchNorm2D(num_features=32)
        self.relu = nn.ReLU()

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        residual = x  # output tensor shape: (1, 1, 28, 28)
        out = self.conv1x1_1(x)  # output tensor shape: (1, 32, 28, 28)
        out = self.bn1(out)  # output tensor shape: (1, 32, 28, 28)
        out = self.relu(out)  # output tensor shape: (1, 32, 28, 28)

        out = self.depthwise_conv(out)  # output tensor shape: (1, 32, 28, 28)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv1x1_2(out)
        out = paddle.add(out, residual)
        out = self.relu(out)
        return out

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Three branches with same block structure
        self.branch1 = nn.Sequential(
            Block(1, 32),
            Block(32, 32)
        )
        self.branch2 = nn.Sequential(
            Block(1, 32),
            Block(32, 32)
        )
        self.branch3 = nn.Sequential(
            Block(1, 32),
            Block(32, 32)
        )

        # Concatenation and classification layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3*32*28*28, 10)  # MNIST images are 28x28

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Split input into three branches
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        # Concatenate outputs
        out = paddle.concat([x1, x2, x3], axis=1)

        # Flatten and classify
        out = self.flatten(out)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, channels, height, width]
    input_tensor = paddle.randn((1, 1, 28, 28))  # MNIST images are 28x28 grayscale
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)