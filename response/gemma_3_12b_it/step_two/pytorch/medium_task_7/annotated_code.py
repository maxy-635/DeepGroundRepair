import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Separate convolutional layer
        self.conv_sep = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels='mask', kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=128)  # Assuming input size after conv layers is 8x8
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Path 1: Through conv1, conv2, and conv3
        x1 = self.conv1(x)  # output tensor shape: (1, 32, 32, 32)
        x1 = self.conv2(x1)  # output tensor shape: (1, 64, 32, 32)
        x1 = self.conv3(x1)  # output tensor shape: (1, 128, 32, 32)

        # Path 2: Through the separate convolutional layer
        x2 = self.conv_sep(x)  # output tensor shape of this layer should be (1, 128, 32, 32)

        # Add the outputs of the two paths
        merged = x1 + x2

        # Flatten the output for fully connected layers
        merged = merged.view(merged.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(merged))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (CIFAR-10 image size: 32x32x3)
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)