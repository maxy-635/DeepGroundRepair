import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2D(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x1 = self.maxpool1(self.conv2(self.conv1(x)))  # output tensor shape: (1, 128, 16, 16)
        x2 = self.maxpool2(self.conv3(x))  # output tensor shape of this layer should be (1, 128, 16, 16)
        x = paddle.add(x1, x2)
        x = self.flatten(x)
        x = self.fc1(x)
        x = paddle.nn.functional.relu(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)