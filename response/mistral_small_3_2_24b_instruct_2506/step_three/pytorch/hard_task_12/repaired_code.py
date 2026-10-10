import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1x1_main = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv1x1_parallel = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3x3_parallel = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Branch path
        self.conv3x3_branch = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32*32*32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # Assuming 10 classes for classification

    def forward(self, x):
        # the initial shape of x is: (1, 64, 32, 32)
        # Main path
        x_main = self.conv1x1_main(x)  # output tensor shape: (1, 32, 32, 32)
        x_parallel1 = self.conv1x1_parallel(x_main)  # output tensor shape: (1, 32, 32, 32)
        x_parallel2 = self.conv3x3_parallel(x_main)  # output tensor shape: (1, 32, 32, 32)
        x_main_concat = torch.cat([x_parallel1, x_parallel2], dim=1)  # output tensor shape: (1, 64, 32, 32)

        # Branch path
        x_branch = self.conv3x3_branch(x)  # output tensor shape of this layer should be (1, 64, 32, 32)

        # Combine main and branch paths
        x_combined = x_main_concat + x_branch

        # Fully connected layers
        x_flat = self.flatten(x_combined)
        x_fc1 = self.fc1(x_flat)
        x_fc2 = self.fc2(x_fc1)

        return x_fc2

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 64, 32, 32)  # Batch size of 1, 64 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])