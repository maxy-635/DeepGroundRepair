import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First path: three sequential convolutional layers
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Second path: direct convolutional layer
        self.conv_direct = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features='mask', out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First path
        x1 = F.relu(self.conv1(x))  # output tensor shape: (1, 64, 32, 32)
        x1 = F.relu(self.conv2(x1))  # output tensor shape: (1, 64, 32, 32)
        x1 = F.relu(self.conv3(x1))  # output tensor shape: (1, 64, 32, 32)

        # Second path
        x2 = F.relu(self.conv_direct(x))  # output tensor shape: (1, 64, 32, 32)

        # Add outputs from both paths
        x_add = paddle.add(x1, x2)  # output tensor shape: (1, 64, 32, 32)

        # Flatten and pass through fully connected layers
        x_flat = paddle.flatten(x_add, start_axis=1)  # output tensor shape: (1, 65536)
        x_fc1 = F.relu(self.fc1(x_flat))
        x_fc2 = self.fc2(x_fc1)

        return x_fc2

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]