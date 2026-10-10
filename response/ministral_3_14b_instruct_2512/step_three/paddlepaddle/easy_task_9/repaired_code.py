import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First 1x1 convolution to increase dimensionality
        self.conv1x1_increase = nn.Conv2D(
            in_channels=1,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # 3x3 depthwise separable convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2D(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=32  # Depthwise convolution
            ),
            nn.ReLU(),
            nn.Conv2D(
                in_channels=32,
                out_channels=32,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )

        # Second 1x1 convolution to reduce dimensionality
        self.conv1x1_reduce = nn.Conv2D(
            in_channels=32,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Flatten and fully connected layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=28*28, out_features=10)  # MNIST images are 28x28

    def forward(self, x):
        # Original input
        original_input = x

        # First 1x1 convolution
        x = self.conv1x1_increase(x)
        x = F.relu(x)

        # Depthwise separable convolution
        x = self.depthwise_conv(x)

        # Second 1x1 convolution
        x = self.conv1x1_reduce(x)

        # Add original input to processed output
        x = x + original_input

        # Flatten and fully connected layer
        x = self.flatten(x)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, channels, height, width]
    # MNIST images are 28x28 grayscale (1 channel)
    input_tensor = paddle.randn((1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # Should be [1, 10]