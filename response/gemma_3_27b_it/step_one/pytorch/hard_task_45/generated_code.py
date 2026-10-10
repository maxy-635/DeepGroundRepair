import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Depthwise Separable Convolutions
        self.block1_group1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        )
        self.block1_group2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        )
        self.block1_group3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
        )

        # Block 2: Multiple Branches
        self.block2_branch1 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=1, stride=1, padding=0)
        )
        self.block2_branch2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        )
        self.block2_branch3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        )
        self.block2_branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=1, stride=1, padding=0)
        )

        # Final Layers
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=32 * 8 * 8, out_features=10)  # Assuming input size 32x32

    def forward(self, x):
        # Block 1
        x_group1 = self.block1_group1(x)
        x_group2 = self.block1_group2(x)
        x_group3 = self.block1_group3(x)
        x = torch.concatenate([x_group1, x_group2, x_group3], dim=1)  # Concatenate along channel dimension

        # Block 2
        x_branch1 = self.block2_branch1(x)
        x_branch2 = self.block2_branch2(x)
        x_branch3 = self.block2_branch3(x)
        x_branch4 = self.block2_branch4(x)
        x = torch.concatenate([x_branch1, x_branch2, x_branch3, x_branch4], dim=1)

        # Final Layers
        x = self.flatten(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)