import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path components
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

        # Branch path component
        self.branch_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Final fully connected layer
        self.fc = nn.Linear(in_features=3*32*32, out_features=10)  # Assuming input size is 32x32 for CIFAR-10

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Split the input into three groups along the last dimension
        x1, x2, x3 = torch.split(x, split_size_or_sections=1, dim=1)  # output tensor shape: (1, 1, 32, 32); (1, 1, 32, 32); (1, 1, 32, 32)

        # Main path processing
        # First group remains unchanged
        # Second group undergoes feature extraction
        x2 = self.conv1(x2)
        # Combine second group with third group
        x_combined = torch.cat([x2, x3], dim=1)
        # Additional convolution
        x_combined = self.conv2(x_combined)
        # Concatenate all three groups
        x_main = torch.cat([x1, x_combined], dim=1)

        # Branch path processing
        x_branch = self.branch_conv(x)

        # Fuse main and branch paths
        x_fused = x_main + x_branch

        # Final classification
        x_flat = torch.flatten(x_fused, start_dim=1)
        x_out = self.fc(x_flat)

        return x_out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)