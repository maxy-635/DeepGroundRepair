import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial Convolution
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.relu1 = nn.ReLU(inplace=True)

        # Block 1
        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels='mask', kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(num_features=64)
        self.relu2_1 = nn.ReLU(inplace=True)

        # Block 2
        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(num_features=64)
        self.relu2_2 = nn.ReLU(inplace=True)

        # Block 3
        self.conv2_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2_3 = nn.BatchNorm2d(num_features=64)
        self.relu2_3 = nn.ReLU(inplace=True)

        # Flatten and Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=128)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Initial Convolution
        x = self.conv1(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.bn1(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.relu1(x)  # output tensor shape: (1, 32, 32, 32)

        # Parallel Blocks
        block1 = self.conv2_1(x)  # output tensor shape of this layer should be (1, 64, 32, 32)
        block1 = self.bn2_1(block1)
        block1 = self.relu2_1(block1)

        block2 = self.conv2_2(x)  # output tensor shape: (1, 64, 32, 32)
        block2 = self.bn2_2(block2)  # output tensor shape: (1, 64, 32, 32)
        block2 = self.relu2_2(block2)  # output tensor shape: (1, 64, 32, 32)

        block3 = self.conv2_3(x)  # output tensor shape: (1, 64, 32, 32)
        block3 = self.bn2_3(block3)  # output tensor shape: (1, 64, 32, 32)
        block3 = self.relu2_3(block3)  # output tensor shape: (1, 64, 32, 32)

        # Add Outputs
        x = x + block1 + block2 + block3

        # Flatten and Fully Connected Layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)