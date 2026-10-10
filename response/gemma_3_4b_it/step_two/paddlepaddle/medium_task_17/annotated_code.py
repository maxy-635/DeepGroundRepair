import paddle
import paddle.vision.datasets as datasets
import paddle.io
import paddle.to_tensor as to_tensor
import paddle.reshape as reshape
import paddle.transpose as transpose
import paddle.nn as nn

class DL_Model(paddle.nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2D(pool_size=2, stride=2)

        self.conv2 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2D(pool_size=2, stride=2)

        self.flatten = nn.AdaptivePool2d(output_size=(1,1))

        self.fc = nn.Linear(32 * 1 * 1, 10)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = reshape(x, (x.shape[1], x.shape[2], 1))
        x = transpose(x, perm=[0, 2, 1])
        x = reshape(x, (x.shape[0], x.shape[2], x.shape[3]))

        x = self.fc(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor)