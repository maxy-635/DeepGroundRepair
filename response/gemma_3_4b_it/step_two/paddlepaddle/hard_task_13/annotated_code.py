import sys
import paddle
import paddle.nn as nn

class DL_Model(paddle.nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Feature Extraction
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2D(in_channels=32, out_channels=48, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)

        # Block 2: Dimensionality Reduction and Weight Multiplication
        self.avg_pool = nn.AvgPool2D(kernel_size=8, stride=1, padding=4)
        self.fc1 = nn.Linear(in_features='mask', out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1
        x1 = self.conv1(x)  # output tensor shape: (1, 16, 32, 32)
        x2 = self.conv2(x1)  # output tensor shape: (1, 32, 32, 32)
        x3 = self.conv3(x2)  # output tensor shape: (1, 48, 32, 32)
        x = self.pool(x3)  # output tensor shape: (1, 48, 16, 16)

        # Block 2
        x = self.avg_pool(x)  # output tensor shape: (1, 48, 17, 17)
        x = x.reshape(x.shape[0], -1)  # output tensor shape: (1, 13872)
        x = self.fc1(x)
        x = self.fc2(x)
        x = paddle.reshape(x, x.shape[0], 3, 3)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (1, 3, 32, 32)
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)