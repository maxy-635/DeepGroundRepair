import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path branches
        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # Branch 2: 1x1 conv followed by 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # Branch 3: 1x1 conv followed by two 3x3 convs
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # 1x1 convolution to adjust dimensions after concatenation
        self.conv1x1 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn1x1 = nn.BatchNorm2d(num_features=64)

        # Direct connection branch
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64)
        )

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(in_features=64*32*32, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=10)  # CIFAR-10 has 10 classes
        )

    def forward(self, x):
        # Main path branches
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)

        # Concatenate outputs from all branches
        concat_out = torch.cat([branch1_out, branch2_out, branch3_out], dim=1)

        # 1x1 convolution to adjust dimensions
        conv_out = self.conv1x1(concat_out)
        conv_out = self.bn1x1(conv_out)

        # Direct connection branch
        shortcut_out = self.shortcut(x)

        # Fuse main path and branch through addition
        fused_out = conv_out + shortcut_out
        fused_out = F.relu(fused_out)

        # Flatten for classification
        fused_out = fused_out.view(fused_out.size(0), -1)

        # Classification
        output = self.classifier(fused_out)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)