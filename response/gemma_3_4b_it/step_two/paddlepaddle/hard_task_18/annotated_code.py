import paddle
import paddle.nn as nn
import paddle.vision.datasets as datasets

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AvgPool2D(kernel_size=8, stride=8)

        # Second block
        self.global_avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.fc1 = nn.Linear(in_features=32 * 1 * 1, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = paddle.nn.functional.relu(x)
        x = self.conv2(x)
        x = paddle.nn.functional.relu(x)
        x = self.avg_pool(x)

        # Second block
        x = self.global_avg_pool(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = paddle.nn.functional.relu(x)
        x = self.fc2(x)
        x = paddle.nn.functional.log_sigmoid(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    input_tensor = paddle.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)