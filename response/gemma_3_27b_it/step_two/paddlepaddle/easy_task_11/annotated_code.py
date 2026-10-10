import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.avg_pool = nn.AvgPool2D(kernel_size=5, stride=3)
        self.conv1d = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features='mask', out_features=128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        x = self.avg_pool(x)  # output tensor shape: (1, 1, 8, 8)
        x = self.conv1d(x)  # output tensor shape: (1, 32, 8, 8)
        x = self.flatten(x)  # output tensor shape: (1, 2048)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)