import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(16)
        self.conv2 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(32)
        self.conv3 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2D(num_features=64)
        self.add_path_conv = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 10 * 10, 128)
        self.bn_fc1 = nn.BatchNorm1D(128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Path 1: Convolutional layers
        x = self.conv1(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.bn1(x)  # output tensor shape: (1, 16, 32, 32)
        x = paddle.nn.functional.relu(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.conv2(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.bn2(x)  # output tensor shape: (1, 32, 32, 32)
        x = paddle.nn.functional.relu(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.conv3(x)  # output tensor shape: (1, 64, 32, 32)
        x = self.bn3(x)
        x = paddle.nn.functional.relu(x)

        # Path 2: Direct convolutional layer
        x_direct = self.add_path_conv(x)

        # Add the outputs
        x = paddle.add(x, x_direct)

        # Flatten and fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = paddle.nn.functional.relu(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)