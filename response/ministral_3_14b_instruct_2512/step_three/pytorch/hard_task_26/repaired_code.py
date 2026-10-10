import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            # Initial 1x1 convolution
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Branch 1: 3x3 convolution
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Branch 2: Downsample with max pooling, 3x3 conv, upsample
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            # Branch 3: Downsample with max pooling, 3x3 conv, upsample
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            # Concatenate all branches and 1x1 conv
            nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(64 * 32 * 32, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        # Main path processing
        main_out = self.main_path[0](x)  # Initial 1x1 conv

        # Branch 1
        branch1 = self.main_path[1:4](main_out)

        # Branch 2
        branch2 = self.main_path[4:10](main_out)

        # Branch 3
        branch3 = self.main_path[10:16](main_out)

        # Concatenate branches
        concat = torch.cat([branch1, branch2, branch3], dim=1)

        # Final 1x1 conv in main path
        main_out = self.main_path[16:](concat)

        # Branch path processing
        branch_out = self.branch_path(x)

        # Add main and branch outputs
        combined = main_out + branch_out

        # Global average pooling
        pooled = F.adaptive_avg_pool2d(combined, (1, 1))

        # Flatten and classify
        flattened = pooled.view(pooled.size(0), -1)
        output = self.classifier(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)