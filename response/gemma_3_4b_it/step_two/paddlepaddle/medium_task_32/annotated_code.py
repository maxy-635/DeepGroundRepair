import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Group 1: 1x1 Conv
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2D(64)
        self.relu1 = nn.ReLU()

        # Group 2: 3x3 Conv
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(64)
        self.relu2 = nn.ReLU()

        # Group 3: 5x5 Conv
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm2D(64)
        self.relu3 = nn.ReLU()

        # Concatenate the three groups
        self.concat = nn.Concat(dim='mask')

        # Flatten
        self.flatten = nn.Flatten()

        # Linear layer
        self.linear = nn.Linear(64 * 3, 10)  # 10 for CIFAR-10

    def forward(self, x):
        # Group 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Group 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # Group 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # Concatenate
        x = self.concat(x)

        # Flatten
        x = self.flatten(x)

        # Linear layer
        x = self.linear(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor for CIFAR-10 (3 channels, 32x32 image)
    input_tensor = paddle.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)