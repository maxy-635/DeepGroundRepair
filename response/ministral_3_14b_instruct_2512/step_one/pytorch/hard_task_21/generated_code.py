import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path with three groups of depthwise separable convolutions
        self.main_path = nn.Sequential(
            # First group: 1x1 depthwise separable convolution
            nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            ),
            # Second group: 3x3 depthwise separable convolution
            nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            ),
            # Third group: 5x5 depthwise separable convolution
            nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2, groups=16),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            )
        )

        # Concatenate the outputs from the three groups
        self.concat = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Branch path with 1x1 convolution
        self.branch_path = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Add the outputs from main and branch paths
        self.add = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Global average pooling and fully connected layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Split the input into three groups along the channel dimension
        x1 = x[:, :1, :, :]  # First group
        x2 = x[:, 1:2, :, :]  # Second group
        x3 = x[:, 2:, :, :]   # Third group

        # Process each group through the main path
        x1 = self.main_path[0](x1)
        x2 = self.main_path[1](x2)
        x3 = self.main_path[2](x3)

        # Concatenate the outputs
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.concat(x)

        # Process the branch path
        branch = self.branch_path(x)

        # Add the outputs from main and branch paths
        x = x + branch
        x = self.add(x)

        # Global average pooling and fully connected layers
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)