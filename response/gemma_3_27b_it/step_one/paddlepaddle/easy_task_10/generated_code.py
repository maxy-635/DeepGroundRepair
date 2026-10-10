import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=8, kernel_size=1, stride=2, padding=0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1, groups=8)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2D(in_channels=8, out_channels=16, kernel_size=1, stride=2, padding=0)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=16 * 7 * 7, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)