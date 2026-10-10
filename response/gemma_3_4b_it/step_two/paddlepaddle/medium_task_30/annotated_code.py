import sys
import paddle
import paddle.nn as nn
import paddle.vision.transforms as transforms
import paddle.io
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        self.conv1 = nn.Conv2D(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pool1 = nn.AdaptiveAvgPool2D((1, 1))
        self.conv2 = nn.Conv2D(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pool2 = nn.AdaptiveAvgPool2D((2, 2))
        self.conv3 = nn.Conv2D(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pool3 = nn.AdaptiveAvgPool2D((4, 4))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)