import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First 1x1 convolution to increase dimensionality
        self.conv1x1_1 = nn.Conv2D(
            in_channels=1,
            out_channels=32,
            kernel_size=1,
            stride=2,
            padding=0
        )

        # 3x3 depthwise separable convolution
        self.depthwise_conv = nn.Sequential(
            nn.Conv2D(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=32  # Depthwise convolution
            ),
            nn.BatchNorm2D(32),
            nn.ReLU()
        )

        # Second 1x1 convolution to reduce dimensionality
        self.conv1x1_2 = nn.Conv2D(
            in_channels=32,
            out_channels=16,
            kernel_size=1,
            stride=2,
            padding=0
        )

        # Adaptive pooling and flattening
        self.pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.flatten = nn.Flatten()

        # Fully connected layer for classification
        self.fc = nn.Linear(in_features=16, out_features=10)

    def forward(self, x):
        x = self.conv1x1_1(x)
        x = self.depthwise_conv(x)
        x = self.conv1x1_2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    # MNIST images are 28x28 with 1 channel
    input_tensor = paddle.randn([1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)