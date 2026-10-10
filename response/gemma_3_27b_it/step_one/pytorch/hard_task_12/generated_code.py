import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.conv1x1_main = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv1x1_parallel = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3x3_parallel = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Branch Path
        self.conv3x3_branch = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Flatten and Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # Assuming 10 classes

    def forward(self, x):
        # Main Path
        x_main = self.conv1x1_main(x)
        x_parallel1 = self.conv1x1_parallel(x_main)
        x_parallel2 = self.conv3x3_parallel(x_main)
        x_main = torch.concatenate([x_parallel1, x_parallel2], dim=1)

        # Branch Path
        x_branch = self.conv3x3_branch(x)

        # Combine Main and Branch Paths
        x = x_main + x_branch

        # Flatten and Fully Connected Layers
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 64, 32, 32)  # Batch size of 1
    output_tensor = model(input_tensor)
    print(output_tensor.shape) # Expected output: torch.Size([1, 10])