import paddle
import paddle.nn as nn
import paddle.vision.datasets as datasets
import paddle.io as pdo
import paddle.optimizer as optim

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Convolutional Block 1
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Convolutional Block 2
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Flatten and Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features='mask', out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)


    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Convolutional Block 1
        x = self.conv1(x)  # output tensor shape: (1, 32, 28, 28)
        x = self.pool1(x)  # output tensor shape: (1, 32, 14, 14)
        x = self.conv2(x)  # output tensor shape: (1, 64, 14, 14)
        x = self.pool1(x)  # output tensor shape: (1, 64, 7, 7)

        # Convolutional Block 2
        x = self.conv3(x)  # output tensor shape: (1, 128, 7, 7)
        x = self.pool2(x)  # output tensor shape: (1, 128, 3, 3)
        x = self.conv4(x)  # output tensor shape: (1, 256, 3, 3)
        x = self.pool2(x)  # output tensor shape: (1, 256, 1, 1)

        # Flatten and Fully Connected Layers
        x = self.flatten(x)  # output tensor shape: (1, 256)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(1, 1, 28, 28)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)