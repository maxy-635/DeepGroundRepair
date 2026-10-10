import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class SpecializedBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(SpecializedBlock, self).__init__()
        self.depthwise_conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels  # This makes it depthwise separable
        )
        self.pointwise_conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        x = self.depthwise_conv(x)  # output tensor shape: (1, 64, 7, 7)
        x = self.dropout1(x)  # output tensor shape: (1, 64, 7, 7)
        x = self.pointwise_conv(x)  # output tensor shape: (1, 128, 7, 7)
        x = self.dropout2(x)  # output tensor shape: (1, 128, 7, 7)
        return x

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Three branches with different input channels
        self.branch1 = SpecializedBlock(in_channels=1, out_channels=32)
        self.branch2 = SpecializedBlock(in_channels=32, out_channels=64)
        self.branch3 = SpecializedBlock(in_channels=64, out_channels=128)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=128*7*7, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Process through three branches
        x1 = self.branch1(x)  # output tensor shape: (1, 32, 28, 28)
        x2 = self.branch2(F.avg_pool2d(x1, kernel_size=2, stride=2))  # output tensor shape: (1, 64, 14, 14)
        x3 = self.branch3(F.avg_pool2d(x2, kernel_size=2, stride=2))  # output tensor shape: (1, 128, 7, 7)

        # Concatenate the outputs
        x = paddle.concat([x1, x2, x3], axis=mask)

        # Flatten and pass through fully connected layers
        x = paddle.flatten(x, start_axis=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    # MNIST images are 28x28 with 1 channel
    input_tensor = paddle.randn((1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)