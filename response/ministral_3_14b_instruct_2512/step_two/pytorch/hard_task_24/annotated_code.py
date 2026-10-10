import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial 1x1 convolutional layer
        self.initial_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Branch 1: 3x3 convolutional layer
        self.branch1_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Branch 2: Max pooling + 3x3 convolution + Upsampling
        self.branch2_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.branch2_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch2_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Branch 3: Max pooling + 3x3 convolution + Upsampling
        self.branch3_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.branch3_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch3_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Fusion layer: 1x1 convolutional layer
        self.fusion_conv = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=10)

        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.bn_fusion = nn.BatchNorm2d(num_features=64)

    def forward(self, x):
        # Initial 1x1 convolution
        x = self.initial_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Branch 1
        branch1 = self.branch1_conv(x)
        branch1 = self.bn2(branch1)
        branch1 = self.relu(branch1)

        # Branch 2
        branch2 = self.branch2_pool(x)
        branch2 = self.branch2_conv(branch2)
        branch2 = self.bn2(branch2)
        branch2 = self.relu(branch2)
        branch2 = self.branch2_upsample(branch2)

        # Branch 3
        branch3 = self.branch3_pool(x)
        branch3 = self.branch3_conv(branch3)
        branch3 = self.bn3(branch3)
        branch3 = self.relu(branch3)
        branch3 = self.branch3_upsample(branch3)

        # Concatenate all branches
        x = torch.cat([branch1, branch2, branch3], dim=1)

        # Fusion layer
        x = self.fusion_conv(x)
        x = self.bn_fusion(x)
        x = self.relu(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)