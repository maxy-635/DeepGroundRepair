import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # First level residual block
        self.level1 = BasicBlock(16, 16)

        # Second level residual blocks
        self.level2_block1 = BasicBlock(16, 32, stride=2)
        self.level2_block2 = BasicBlock(32, 32)

        # Third level global branch
        self.global_branch = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(32)
        )

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        # Initial feature extraction
        x = F.relu(self.bn1(self.conv1(x)))

        # First level residual connection
        x = self.level1(x)

        # Second level residual connection
        x_level2 = self.level2_block1(x)
        x_level2 = self.level2_block2(x_level2)

        # Third level global branch
        global_branch = self.global_branch(x)

        # Combine features
        x = x_level2 + global_branch

        # Final classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # Should be [1, 10]