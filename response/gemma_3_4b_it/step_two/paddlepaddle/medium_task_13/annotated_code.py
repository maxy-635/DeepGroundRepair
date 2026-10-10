import paddle
import paddle.vision.datasets as datasets
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features='mask', out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x = self.conv1(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.softmax(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.conv2(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.softmax(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.conv3(x)  # output tensor shape: (1, 64, 32, 32)
        x = self.softmax(x)  # output tensor shape: (1, 64, 32, 32)
        x = self.flatten(x)  # output tensor shape: (1, 65536)
        x = self.fc1(x)
        x = self.softmax(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)