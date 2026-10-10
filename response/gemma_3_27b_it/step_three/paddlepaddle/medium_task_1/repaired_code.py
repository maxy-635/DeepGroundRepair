import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=16384, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x = self.conv1(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.conv2(x)  # output tensor shape: (1, 64, 32, 32)
        x = self.pool(x)  # output tensor shape: (1, 64, 16, 16)
        x = x + x  # output tensor shape: (1, 64, 16, 16)
        x = self.flatten(x)  # output tensor shape: (1, 16384)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)