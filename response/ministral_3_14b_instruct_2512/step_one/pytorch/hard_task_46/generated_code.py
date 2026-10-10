import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Split channels and apply separable convolutions with different kernel sizes
        self.split_conv_block = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),  # Split into 3 groups (1 channel each)
            nn.ReLU(inplace=True),

            # Group 1: 1x1 convolution
            nn.Conv2d(1, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Group 2: 3x3 convolution
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Group 3: 5x5 convolution
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Concatenate the three groups
            nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0),  # 64*3=192 input channels
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Second block: Multiple branches for enhanced feature extraction
        self.branch_block = nn.Sequential(
            # Branch 1: 3x3 convolution
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Branch 2: 1x1 + 3x3 + 3x3 convolutions
            nn.Sequential(
                nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),

            # Branch 3: Max pooling
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # Concatenate all branches
            nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0),  # 64*3=192 input channels
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Global average pooling and final classification
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)  # 10 classes for CIFAR-10

    def forward(self, x):
        # First block processing
        x = self.split_conv_block(x)

        # Second block processing
        # Branch 1
        branch1 = self.branch_block[0](x)
        # Branch 2
        branch2 = self.branch_block[1](x)
        # Branch 3
        branch3 = self.branch_block[2](x)

        # Concatenate branches
        x = torch.cat([branch1, branch2, branch3], dim=1)
        x = self.branch_block[3](x)

        # Global average pooling and classification
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)