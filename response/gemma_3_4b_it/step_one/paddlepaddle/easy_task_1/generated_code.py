import paddle
import paddle.vision.datasets as datasets
from paddle.nn import Conv2D, MaxPool2D, Flatten, Linear, Sequential, Softmax2D

class DL_Model(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = MaxPool2D(kernel_size=2, stride=2)
        self.flatten = Flatten()
        self.fc1 = Linear(in_features=64 * 7 * 7, out_features=128)
        self.fc2 = Linear(in_features=128, out_features=10)
        self.softmax = Softmax2D()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    input_tensor = paddle.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)