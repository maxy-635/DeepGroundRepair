import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # 1x1 convolutional layer to increase dimensionality
        self.conv1x1_expand = nn.Conv2D(
            in_channels=1,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding=0
        )
        # 3x3 depthwise separable convolutional layer for feature extraction
        self.depthwise_conv = nn.Conv2D(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=32
        )
        # 1x1 convolutional layer to reduce dimensionality
        self.conv1x1_reduce = nn.Conv2D(
            in_channels=32,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )
        # Flattening layer
        self.flatten = nn.Flatten()
        # Fully connected layer
        self.fc = nn.Linear(
            in_features=28*28,
            out_features=10
        )

    def forward(self, x):
        # Save the original input
        original_input = x
        # 1x1 convolution to expand dimensions
        x = self.conv1x1_expand(x)
        # Depthwise separable convolution
        x = self.depthwise_conv(x)
        # 1x1 convolution to reduce dimensions
        x = self.conv1x1_reduce(x)
        # Add the original input to the processed output
        x = x + original_input
        # Flatten and pass through fully connected layer
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])  # batch_size=1, channels=1, height=28, width=28
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # Expected output shape: [1, 10]