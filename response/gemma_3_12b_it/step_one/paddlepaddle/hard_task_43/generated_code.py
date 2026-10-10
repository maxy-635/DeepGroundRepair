import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.avg_pool1 = nn.AvgPool2D(kernel_size=1, stride=1, padding=0)
        self.avg_pool2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.avg_pool3 = nn.AvgPool2D(kernel_size=4, stride=4, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=1, out_features=1024, bias=True)
        self.reshape = nn.Reshape((1024, 1, 1, 1))

        # Block 2
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=1024, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels=1024, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3),
            nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2D(kernel_size=2, stride=2, padding=0),
            nn.Conv2D(in_channels=1024, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        )

        self.fc2 = nn.Linear(in_features=128 * 3, out_features=128, bias=True)
        self.fc3 = nn.Linear(in_features=128, out_features=10, bias=True)

    def forward(self, x):
        # Block 1
        x1 = self.avg_pool1(x)
        x2 = self.avg_pool2(x)
        x3 = self.avg_pool3(x)

        x1 = self.flatten(x1)
        x2 = self.flatten(x2)
        x3 = self.flatten(x3)

        x = paddle.concat([x1, x2, x3], axis=-1)
        x = self.fc1(x)
        x = self.reshape(x)
        # Block 2
        x = self.branch1(x)
        x = self.branch2(x)
        x = self.branch3(x)

        x = paddle.concat([x], axis=1)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.rand(shape=[1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)