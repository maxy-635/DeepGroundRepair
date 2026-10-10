import paddle
import paddle.vision.datasets as datasets
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.AvgPool2D(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.AvgPool2D(kernel_size=2, stride=2)

        # Path 2
        self.conv3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Combine features
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(in_features='mask', out_features=128)
        self.celu1 = nn.CELU(alpha=1.0)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.softmax2d = nn.Softmax(axis=-1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        path1 = x  # output tensor shape: (1, 3, 32, 32)
        path1 = self.conv1(path1)  # output tensor shape: (1, 32, 32, 32)
        path1 = self.pool1(path1)  # output tensor shape: (1, 32, 16, 16)
        path1 = self.conv2(path1)  # output tensor shape: (1, 64, 16, 16)
        path1 = self.pool2(path1)  # output tensor shape: (1, 64, 8, 8)

        path2 = self.conv3(x)  # output tensor shape: (1, 64, 32, 32)

        path1 = self.flatten1(path1)  # output tensor shape: (1, 4096)
        path2 = self.flatten1(path2)  # output tensor shape: (1, 65536)

        combined_features = paddle.concat((path1, path2), axis=1)  # output tensor shape: (1, 69632)
        combined_features = self.fc1(combined_features)
        combined_features = self.celu1(combined_features)
        output = self.fc2(combined_features)
        output = self.softmax2d(output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (batch_size, channels, height, width)
    input_tensor = paddle.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)