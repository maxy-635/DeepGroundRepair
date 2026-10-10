import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # 1x1 convolutional layer to increase dimensionality
        self.conv1x1_inc = nn.Conv2D(
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
            padding=1
        )
        # 1x1 convolutional layer to reduce dimensionality
        self.conv1x1_dec = nn.Conv2D(
            in_channels=32,
            out_channels=16,
            kernel_size=1,
            stride=1,
            padding=0
        )
        # ReLU activation function
        self.relu = nn.ReLU()
        # Flatten layer
        self.flatten = nn.Flatten()
        # Fully connected layer for classification
        self.linear = nn.Linear(
            in_features=16 * 7 * 7,
            out_features=10
        )

    def forward(self, x):
        # 1x1 convolution
        x = self.conv1x1_inc(x)
        # Depthwise separable convolution
        x = self.depthwise_conv(x)
        # 1x1 convolution
        x = self.conv1x1_dec(x)
        # Add the processed output to the original input
        x = x + x
        # ReLU activation
        x = self.relu(x)
        # Flatten the output
        x = self.flatten(x)
        # Fully connected layer
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for MNIST (1, 1, 28, 28)
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)