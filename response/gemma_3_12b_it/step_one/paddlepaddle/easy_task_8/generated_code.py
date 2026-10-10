import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Depthwise separable convolution layer
        self.depthwise_conv = nn.Conv2D(
            in_channels=1,  # MNIST grayscale images have 1 channel
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.depthwise_relu = nn.ReLU()
        self.depthwise_dropout = nn.Dropout(p=0.1)

        # 1x1 convolutional layer
        self.pointwise_conv = nn.Conv2D(
            in_channels=32,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.pointwise_relu = nn.ReLU()
        self.pointwise_dropout = nn.Dropout(p=0.1)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layer
        self.linear = nn.Linear(
            in_features=64 * 7 * 7,  # Assuming input size after conv layers is 28x28 -> 22x22 -> 20x20 -> 14x14 -> 7x7, and 64 channels
            out_features=10  # 10 classes for MNIST
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.depthwise_relu(x)
        x = self.depthwise_dropout(x)

        x = self.pointwise_conv(x)
        x = self.pointwise_relu(x)
        x = self.pointwise_dropout(x)

        x = self.flatten(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 1, 28, 28))  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)