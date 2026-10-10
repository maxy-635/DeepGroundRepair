import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=8 * 28 * 28, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + x  # Add processed output to original input
        x = self.flatten(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.to_tensor(paddle.randn(1, 1, 28, 28), dtype=paddle.float32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)