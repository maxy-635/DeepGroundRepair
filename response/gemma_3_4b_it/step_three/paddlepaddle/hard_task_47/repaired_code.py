import paddle
import paddle.nn as nn
import paddle.vision.datasets as datasets
import paddle.io
import paddle.vision.transforms as transforms

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Depthwise Separable Convolution
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2D(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(16)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2D(16)
        self.relu3 = nn.ReLU()
        self.concat1 = nn.Concat(dim=0)

        # Block 2: Feature Extraction Branches
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(32),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(32),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2D(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(32),
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2D(32),
            nn.ReLU(),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(32),
            nn.ReLU()
        )
        self.avgpool = nn.AvgPool2D(kernel_size=8, stride=1, padding=4)
        self.concat2 = nn.Concat(dim=1)

        # Classification Layers
        self.fc1 = nn.Linear(32 * 3, 128)
        self.bn_fc1 = nn.BatchNorm1D(128)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Block 1
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

        # Block 2
        x = self.branch1(x)
        x = self.branch2(x)
        x = self.branch3(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)  # Flatten
        x = self.concat2([x, x, x])

        # Classification
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu_fc1(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    batch_size = 4
    channels = 3
    height = 32
    width = 32
    input_tensor = paddle.randn((batch_size, channels, height, width))
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)