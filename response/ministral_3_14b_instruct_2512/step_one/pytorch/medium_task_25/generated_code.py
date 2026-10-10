import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1: Single 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Path 2: Average pooling followed by 1x1 convolution
        self.path2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Path 3: 1x1 conv followed by parallel 1x3 and 3x1 convolutions
        self.path3 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        )

        # Path 4: 1x1 conv followed by 3x3 conv, then parallel 1x3 and 3x1 convolutions
        self.path4 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        )

        # Global average pooling to reduce spatial dimensions
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer for classification
        self.fc = nn.Linear(256, 10)  # 4 paths * 64 channels = 256 features

    def forward(self, x):
        # Path 1
        out1 = self.path1(x)

        # Path 2
        out2 = self.path2(x)

        # Path 3
        x3 = self.path3[0:6](x)  # First 6 layers
        conv1x3 = self.path3[6](x3)
        conv3x1 = self.path3[8](x3)
        out3 = torch.cat([conv1x3, conv3x1], dim=1)
        out3 = self.path3[9](out3)

        # Path 4
        x4 = self.path4[0:6](x)  # First 6 layers
        conv1x3 = self.path4[6](x4)
        conv3x1 = self.path4[8](x4)
        out4 = torch.cat([conv1x3, conv3x1], dim=1)
        out4 = self.path4[9](out4)

        # Concatenate all paths
        out = torch.cat([out1, out2, out3, out4], dim=1)

        # Global average pooling
        out = self.avgpool(out)

        # Flatten and pass through fully connected layer
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)