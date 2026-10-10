import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the three convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Define the direct convolutional layer
        self.direct_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Define ReLU and MaxPool layers
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Define fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # Path 1: First two convolutional layers
        x1 = self.relu(self.conv1(x))
        x1 = self.maxpool(x1)
        x1 = self.relu(self.conv2(x1))
        x1 = self.maxpool(x1)

        # Path 2: Third convolutional layer
        x2 = self.relu(self.conv3(x1))

        # Add the outputs of the first two layers and the third layer
        x_add = torch.add(x1, x2)

        # Direct path
        x_direct = self.relu(self.direct_conv(x))
        x_direct = self.maxpool(x_direct)

        # Combine all paths
        combined = torch.add(x_add, x_direct)

        # Flatten and pass through fully connected layers
        combined = combined.view(combined.size(0), -1)
        x = self.relu(self.fc1(combined))
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)