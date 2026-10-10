import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1x1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)

        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.bn3 = nn.BatchNorm2d(num_features=64)

        self.relu = nn.ReLU(inplace=True)

        # Branch path
        self.branch_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch_bn = nn.BatchNorm2d(num_features=64)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

        # Adaptive average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(8, 8))

    def forward(self, x):
        # Main path
        x1 = self.conv1x1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv3x3(x)
        x2 = self.bn2(x2)
        x2 = self.relu(x2)

        x3 = self.conv5x5(x)
        x3 = self.bn3(x3)
        x3 = self.relu(x3)

        # Concatenate the outputs from the main path
        x_main = torch.cat([x1, x2, x3], dim=1)

        # Branch path
        x_branch = self.branch_conv(x)
        x_branch = self.branch_bn(x_branch)
        x_branch = self.relu(x_branch)

        # Combine main and branch paths
        x_fused = x_main + x_branch

        # Adaptive average pooling
        x_pool = self.avg_pool(x_fused)

        # Flatten the tensor
        x_flat = x_pool.view(x_pool.size(0), -1)

        # Fully connected layers
        x_fc1 = self.fc1(x_flat)
        x_fc1 = self.relu(x_fc1)
        x_fc2 = self.fc2(x_fc1)

        return x_fc2

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])