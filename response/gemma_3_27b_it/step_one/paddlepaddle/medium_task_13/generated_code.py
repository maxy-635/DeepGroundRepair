import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=64 * 32 * 32, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=10)
        self.softmax = nn.Softmax(axis=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = paddle.concat([x, self.conv1(x)], axis=1)
        x = self.relu(self.conv2(x))
        x = paddle.concat([x, self.conv2(x)], axis=1)
        x = self.relu(self.conv3(x))
        x = paddle.concat([x, self.conv3(x)], axis=1)
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)