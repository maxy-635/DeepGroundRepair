import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path with three convolutional layers
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Direct path convolutional layer
        self.direct_conv = nn.Conv2D(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=128*8*8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

        # Activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path
        x1 = self.relu(self.conv1(x))  # output tensor shape: (1, 64, 32, 32)
        x2 = self.relu(self.conv2(x1))  # output tensor shape: (1, 128, 32, 32)
        x3 = self.relu(self.conv3(x2))  # output tensor shape: (1, 128, 32, 32)

        # Direct path
        direct = self.relu(self.direct_conv(x))  # output tensor shape: (1, 128, 32, 32)

        # Add outputs from main path and direct path
        combined = x3 + direct  # output tensor shape: (1, 128, 32, 32)

        # Flatten for fully connected layers
        combined = paddle.flatten(combined, start_axis=1)  # output tensor shape: (1, 131072)

        # Fully connected layers
        x = self.relu(self.fc1(combined))
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)