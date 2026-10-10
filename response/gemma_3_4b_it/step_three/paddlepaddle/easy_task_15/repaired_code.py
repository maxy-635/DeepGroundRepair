import paddle
import paddle.vision.datasets as datasets
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        self.conv1 = nn.Conv2D(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn1 = nn.BatchNorm2D(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2D(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2D(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2D(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2D(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2D(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn3 = nn.BatchNorm2D(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AvgPool2D(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=1152, out_features=512)
        self.bn4 = nn.BatchNorm2D(512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)
        self.celu = nn.CELU(alpha=1.0)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        x = self.conv1(x)  # output tensor shape: (1, 32, 28, 28)
        x = self.bn1(x)  # output tensor shape: (1, 32, 28, 28)
        x = self.relu1(x)  # output tensor shape: (1, 32, 28, 28)
        x = self.pool1(x)  # output tensor shape: (1, 32, 14, 14)

        x = self.conv2(x)  # output tensor shape: (1, 64, 14, 14)
        x = self.bn2(x)  # output tensor shape: (1, 64, 14, 14)
        x = self.relu2(x)  # output tensor shape: (1, 64, 14, 14)
        x = self.pool2(x)  # output tensor shape: (1, 64, 7, 7)

        x = self.conv3(x)  # output tensor shape: (1, 128, 7, 7)
        x = self.bn3(x)  # output tensor shape: (1, 128, 7, 7)
        x = self.relu3(x)  # output tensor shape: (1, 128, 7, 7)
        x = self.pool3(x)  # output tensor shape: (1, 128, 3, 3)

        x = self.dropout(x)  # output tensor shape: (1, 128, 3, 3)
        x = self.flatten(x)  # output tensor shape: (1, 1152)
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.fc2(x)
        x = self.celu(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)