import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path with three branches of different kernel sizes
        self.main_path = nn.Sequential(
            # First group (1x1 convolution)
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True)
            ),
            # Second group (3x3 convolution)
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True)
            ),
            # Third group (5x5 convolution)
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True)
            )
        )

        # Branch path with 1x1 convolution
        self.branch_path = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=192, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=192, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        # Main path processing - process the entire input through each branch
        main_out1 = self.main_path[0](x)
        main_out2 = self.main_path[1](x)
        main_out3 = self.main_path[2](x)

        # Concatenate the outputs from the three branches
        main_out = torch.cat([main_out1, main_out2, main_out3], dim=1)

        # Branch path processing
        branch_out = self.branch_path(x)

        # Fusion
        fused = main_out + branch_out
        fused = self.fusion(fused)

        # Classification
        output = self.classifier(fused)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)