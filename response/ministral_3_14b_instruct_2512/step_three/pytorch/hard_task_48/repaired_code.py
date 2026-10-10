import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three parallel branches with different kernel sizes
        self.block1_branch1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64)
        )

        self.block1_branch2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64)
        )

        self.block1_branch3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=64)
        )

        # Concatenate the outputs of the three branches
        self.block1_concat = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=128)
        )

        # Block 2: Four parallel branches
        # Path 1: 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64)
        )

        # Path 2: 3x3 average pooling + 1x1 convolution
        self.path2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64)
        )

        # Path 3: 1x1 conv + (1x3 + 3x1) convolutions
        self.path3_conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.path3_conv1x3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.path3_conv3x1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.path3_concat = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Path 4: 1x1 conv + 3x3 conv + (1x3 + 3x1) convolutions
        self.path4_conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.path4_conv3x3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.path4_conv1x3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.path4_conv3x1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.path4_concat = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Final classification layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=64*4, out_features=10)  # 64*4 from concatenating all paths

    def forward(self, x):
        # Block 1 processing
        branch1 = self.block1_branch1(x)
        branch2 = self.block1_branch2(x)
        branch3 = self.block1_branch3(x)

        # Concatenate along channel dimension
        concat = torch.cat([branch1, branch2, branch3], dim=1)
        block1_out = self.block1_concat(concat)

        # Block 2 processing
        # Path 1
        path1 = self.path1(block1_out)

        # Path 2
        path2 = self.path2(block1_out)

        # Path 3
        path3_conv1 = self.path3_conv1(block1_out)
        path3_conv1x3 = self.path3_conv1x3(path3_conv1)
        path3_conv3x1 = self.path3_conv3x1(path3_conv1)
        path3_concat = torch.cat([path3_conv1x3, path3_conv3x1], dim=1)
        path3 = self.path3_concat(path3_concat)

        # Path 4
        path4_conv1 = self.path4_conv1(block1_out)
        path4_conv3x3 = self.path4_conv3x3(path4_conv1)
        path4_conv1x3 = self.path4_conv1x3(path4_conv3x3)
        path4_conv3x1 = self.path4_conv3x1(path4_conv3x3)
        path4_concat = torch.cat([path4_conv1x3, path4_conv3x1], dim=1)
        path4 = self.path4_concat(path4_concat)

        # Concatenate all paths
        all_paths = torch.cat([path1, path2, path3, path4], dim=1)

        # Final classification
        x = self.avgpool(all_paths)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # Should be [1, 10] for 10 classes