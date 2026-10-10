import sys
import paddle
import paddle.nn as nn
import paddle.vision.datasets as datasets
import paddle.io

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2D(num_features=16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(num_features=32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2D(num_features=64)
        self.relu3 = nn.ReLU()
        self.concat1 = nn.Concat(dim='mask')

        # Second block
        self.conv4 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2D(num_features=128)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2D(num_features=128)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2D(num_features=128)
        self.relu6 = nn.ReLU()
        self.maxpool = nn.MaxPool2D(pool_size=2, stride=2)
        self.conv7 = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.bn7 = nn.BatchNorm2D(num_features=128)
        self.relu7 = nn.ReLU()
        self.concat2 = nn.Concat(dim=1)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=128 * 8 * 8, out_features=10)  # Adjust the input size based on the output size of the second block
        self.celu = nn.CELU(alpha=1.0)


    def forward(self, x):
        # First block
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)
        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = self.relu3(x3)
        x = self.concat1([x1, x2, x3])

        # Second block
        x4 = self.conv4(x)
        x4 = self.bn4(x4)
        x4 = self.relu4(x4)
        x5 = self.conv5(x4)
        x5 = self.bn5(x5)
        x5 = self.relu5(x5)
        x6 = self.conv6(x5)
        x6 = self.bn6(x6)
        x6 = self.relu6(x6)
        x = self.maxpool(x6)
        x7 = self.conv7(x)
        x7 = self.bn7(x7)
        x7 = self.relu7(x7)
        x = self.concat2([x4, x5, x7])

        x = self.flatten(x)
        x = self.fc(x)
        x = self.celu(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)