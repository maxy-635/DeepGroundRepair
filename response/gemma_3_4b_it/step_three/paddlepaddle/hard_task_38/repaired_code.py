import paddle
import paddle.vision.datasets as datasets
import paddle.io as pio
import paddle.nn.functional as F
import paddle.nn as nn
import paddle.optimizer as optim

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First pathway
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(64)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2D(128)
        self.relu3 = nn.ReLU()

        # Second pathway
        self.conv4 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2D(32)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2D(64)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2D(128)
        self.relu6 = nn.ReLU()

        # Merge pathway outputs
        self.concat = nn.Concat(dim=1)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 2, 512)
        self.bn_fc1 = nn.BatchNorm1D(512)
        self.relu_fc1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # First pathway
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = F.relu(self.bn3(self.conv3(x1)))

        # Second pathway
        x2 = F.relu(self.bn4(self.conv4(x)))
        x2 = F.relu(self.bn5(self.conv5(x2)))
        x2 = F.relu(self.bn6(self.conv6(x2)))

        # Merge pathway outputs
        x = self.concat(x1, x2)

        # Flatten
        x = x.flatten(1)

        # Fully connected layers
        x = F.normalize(x, p=2, axis=1)
        x = self.relu_fc1(self.bn_fc1(self.fc1(x)))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)