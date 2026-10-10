import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1x1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2D(num_features=64)
        self.bn2 = nn.BatchNorm2D(num_features=64)
        self.bn3 = nn.BatchNorm2D(num_features=64)
        self.relu = nn.ReLU()

        # Branch path
        self.branch_conv = nn.Conv2D(in_channels=3, out_channels=192, kernel_size=1, stride=1, padding=0)
        self.branch_bn = nn.BatchNorm2D(num_features=64)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path
        x1 = self.conv1x1(x)  # output tensor shape: (1, 64, 32, 32)
        x1 = self.bn1(x1)  # output tensor shape: (1, 64, 32, 32)
        x1 = self.relu(x1)  # output tensor shape: (1, 64, 32, 32)

        x2 = self.conv3x3(x)  # output tensor shape: (1, 64, 32, 32)
        x2 = self.bn2(x2)  # output tensor shape: (1, 64, 32, 32)
        x2 = self.relu(x2)  # output tensor shape: (1, 64, 32, 32)

        x3 = self.conv5x5(x)  # output tensor shape: (1, 64, 32, 32)
        x3 = self.bn3(x3)  # output tensor shape: (1, 64, 32, 32)
        x3 = self.relu(x3)  # output tensor shape: (1, 64, 32, 32)

        # Concatenate the outputs from the main path
        x_main = paddle.concat([x1, x2, x3], axis=1)  # output tensor shape: (1, 192, 32, 32)

        # Branch path
        x_branch = self.branch_conv(x)  # output tensor shape of this layer should be (1, 192, 32, 32)
        x_branch = self.branch_bn(x_branch)

        # Combine main and branch paths
        x_fused = paddle.add(x_main, x_branch)

        # Flatten the fused features
        x_flat = paddle.flatten(x_fused, start_axis=1, stop_axis=-1)

        # Fully connected layers
        x_fc1 = self.fc1(x_flat)
        x_fc1 = self.relu(x_fc1)
        x_fc2 = self.fc2(x_fc1)

        return x_fc2

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)