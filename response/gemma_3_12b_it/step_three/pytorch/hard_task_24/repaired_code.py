import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial 1x1 convolutional layer
        self.initial_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Branch 1: Local feature extraction
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Branch 2: Downsampling, convolution, and upsampling
        self.branch2_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.branch2_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch2_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Branch 3: Downsampling, convolution, and upsampling
        self.branch3_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.branch3_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch3_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Fusion layer
        self.fusion_conv = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)

        # Branch 1
        branch1_out = self.branch1(x)

        # Branch 2
        branch2_out = self.branch2_pool(x)
        branch2_out = self.branch2_conv(branch2_out)
        branch2_out = self.branch2_upsample(branch2_out)

        # Branch 3
        branch3_out = self.branch3_pool(x)
        branch3_out = self.branch3_conv(branch3_out)
        branch3_out = self.branch3_upsample(branch3_out)

        # Concatenation
        merged_out = torch.cat((branch1_out, branch2_out, branch3_out), dim=1)

        # Fusion
        merged_out = self.fusion_conv(merged_out)

        # Flatten for fully connected layers
        merged_out = merged_out.view(merged_out.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(merged_out))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)