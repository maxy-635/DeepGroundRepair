import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1 - Primary path
        self.primary_path = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, groups=16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0)
        )

        # Block 1 - Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, groups=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0)
        )

        # Channel shuffling (Block 2)
        self.channel_shuffle = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # To get the shape information
            nn.Flatten()
        )

        # Final classification layer
        self.fc = nn.Linear(in_features=32*28*28, out_features=10)  # MNIST has 10 classes

    def forward(self, x):
        # Block 1 - Primary path
        primary_features = self.primary_path(x)

        # Block 1 - Branch path
        branch_features = self.branch_path(x)

        # Concatenate along channel dimension
        combined_features = torch.cat([primary_features, branch_features], dim=1)

        # Channel shuffling (Block 2)
        # Get the shape information
        batch_size, channels, height, width = combined_features.shape
        groups = 4
        channels_per_group = channels // groups

        # Reshape and permute for channel shuffling
        reshaped = combined_features.view(batch_size, height, width, channels)
        reshaped = reshaped.permute(0, 3, 1, 2)  # (batch, channels, height, width) -> (batch, height, width, channels)
        reshaped = reshaped.view(batch_size, height, width, groups, channels_per_group)
        reshaped = reshaped.permute(0, 1, 2, 4, 3)  # Swap last two dimensions
        reshaped = reshaped.view(batch_size, height, width, channels)
        reshaped = reshaped.permute(0, 3, 1, 2)  # Back to (batch, channels, height, width)

        # Flatten and classify
        flattened = reshaped.view(reshaped.size(0), -1)
        output = self.fc(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(32, 1, 28, 28)  # Batch of 32 MNIST images (1 channel, 28x28)
    output_tensor = model(input_tensor)

    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)