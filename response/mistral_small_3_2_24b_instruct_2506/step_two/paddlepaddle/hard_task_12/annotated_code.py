import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1x1_reduce = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Branch path
        self.branch_conv = nn.Conv2D(in_channels=64, out_channels='mask', kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32*32*32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # Assuming 10 classes for classification

    def forward(self, x):
        # the initial shape of x is: (1, 64, 32, 32)
        # Main path
        x_main = self.conv1x1_reduce(x)  # output tensor shape: (1, 32, 32, 32)
        x1x1 = self.conv1x1(x_main)  # output tensor shape: (1, 32, 32, 32)
        x3x3 = self.conv3x3(x_main)  # output tensor shape: (1, 32, 32, 32)
        x_main = paddle.concat([x1x1, x3x3], axis=1)  # output tensor shape: (1, 64, 32, 32)

        # Branch path
        x_branch = self.branch_conv(x)  # output tensor shape of this layer should be (1, 64, 32, 32)

        # Combine main and branch paths
        x_combined = paddle.add(x_main, x_branch)

        # Fully connected layers
        x_flat = self.flatten(x_combined)
        x_fc1 = F.relu(self.fc1(x_flat))
        x_fc2 = self.fc2(x_fc1)

        return x_fc2

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 64, 32, 32))  # Shape: (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)