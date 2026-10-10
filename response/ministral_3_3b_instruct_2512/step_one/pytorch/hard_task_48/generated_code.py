import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Separable Convolutional Layers with Different Kernel Sizes
        self.block1_conv1x1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64)
        )
        self.block1_conv3x3 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.block1_conv5x5 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64)
        )

        # Concatenate outputs of the three groups
        self.block1_concat = nn.Sequential(
            nn.Conv2d(3 * 64, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128)
        )

        # Block 2: Four Parallel Branches
        self.path1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        )

        self.path2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        )

        self.path3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 128, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.Conv2d(128, 128, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )

        self.path4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 128, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.Conv2d(128, 128, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )

        # Concatenate outputs of the four paths
        self.block2_concat = nn.Sequential(
            nn.Conv2d(4 * 128, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256)
        )

        # Final classification layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 32 * 32, 10)

    def forward(self, x):
        # Block 1: Separable Convolutional Layers
        x1 = self.block1_conv1x1(x)
        x2 = self.block1_conv3x3(x)
        x3 = self.block1_conv5x5(x)

        # Concatenate outputs
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.block1_concat(x)

        # Block 2: Four Parallel Branches
        path1_out = self.path1(x)
        path2_out = self.path2(x)
        path3_out = self.path3(x)
        path4_out = self.path4(x)

        # Concatenate outputs of the four paths
        x = torch.cat((path1_out, path2_out, path3_out, path4_out), dim=1)
        x = self.block2_concat(x)

        # Final classification
        x = self.flatten(x)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor with batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (1, 10)