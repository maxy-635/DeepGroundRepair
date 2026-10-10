import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Increase the dimensionality of the input's channels threefold with a 1x1 convolution
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=9, kernel_size=1, stride=1, padding=0)

        # Extract initial features using a 3x3 depthwise separable convolution
        self.depthwise_conv = nn.Conv2D(in_channels=9, out_channels=9, kernel_size=3, stride=1, padding=1, groups=9)

        # Compute channel attention weights
        self.avg_pool = nn.AvgPool2D(kernel_size=8, stride=8, padding=0)
        self.fc1 = nn.Linear(in_features=9, out_features=9)
        self.fc2 = nn.Linear(9, 9)

        # Reduce dimensionality with a 1x1 convolution
        self.conv2 = nn.Conv2D(in_channels=9, out_channels=9, kernel_size=1, stride=1, padding=0)

        # Combine with the initial input
        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear(9 * 8 * 8, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Increase the dimensionality of the input's channels threefold with a 1x1 convolution
        x = self.conv1(x)  # output tensor shape: (1, 9, 32, 32)

        # Extract initial features using a 3x3 depthwise separable convolution
        x = self.depthwise_conv(x)  # output tensor shape: (1, 9, 32, 32)

        # Compute channel attention weights
        x = self.avg_pool(x)  # output tensor shape: (1, 9, 4, 4)
        x = self.fc1(x)
        x = x.reshape(x.shape[0], 9)
        x = self.fc2(x)
        x = x.reshape(x.shape[0], 9)
        x = x * x

        # Reduce dimensionality with a 1x1 convolution
        x = self.conv2(x)

        # Combine with the initial input
        x = self.flatten(x)
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)