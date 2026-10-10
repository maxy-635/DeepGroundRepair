import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path block definition
        class MainPathBlock(nn.Module):
            def __init__(self, in_channels, out_channels):
                super(MainPathBlock, self).__init__()
                self.sep_conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=in_channels),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )

            def forward(self, x):
                # Get the input features
                input_features = x
                # Apply separable convolution
                conv_features = self.sep_conv(x)
                # Concatenate along channel dimension
                return torch.cat([input_features, conv_features], dim=1)

        # Branch path
        self.branch_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Main path with 3 blocks
        self.main_path = nn.Sequential(
            MainPathBlock(1, 32),
            MainPathBlock(64, 32),
            MainPathBlock(96, 32)
        )

        # Final layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(64 * 28 * 28, 10)  # MNIST images are 28x28

    def forward(self, x):
        # Main path processing
        x = self.main_path(x)

        # Branch path processing
        branch_features = self.branch_conv(x)

        # Feature fusion
        fused_features = x + branch_features

        # Final processing
        x = self.final_conv(fused_features)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch of 1 MNIST image (1 channel, 28x28)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # Should be [1, 10]