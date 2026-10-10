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
        # the initial shape of x is: (32, 1, 28, 28)
        # Block 1 - Primary path
        primary_features = self.primary_path(x)  # output tensor shape: (32, 32, 28, 28)

        # Block 1 - Branch path
        branch_features = self.branch_path(x)  # output tensor shape: (32, 32, 28, 28)

        # Concatenate along channel dimension
        combined_features = torch.cat([primary_features, branch_features], dim=1)  # output tensor shape: (32, 64, 28, 28)

        # Channel shuffling (Block 2)
        # Get the shape information
        batch_size, channels, height, width = combined_features.shape  # output tensor shape: (1); (1); (1); (1)
        groups = 4  # output tensor shape: (1)
        channels_per_group = channels // groups  # output tensor shape: (1)

        # Reshape and permute for channel shuffling
        reshaped = combined_features.view(batch_size, height, width, channels)  # output tensor shape: (32, 28, 28, 64)
        reshaped = reshaped.permute(0, 3, 1, 2)  # output tensor shape: (32, 64, 28, 28)
        reshaped = reshaped.view('mask', 'mask', 'mask', 'mask', 'mask')
        reshaped = reshaped.permute(0, 1, 2, 4, 3)
        reshaped = reshaped.view(batch_size, height, width, channels)
        reshaped = reshaped.permute(0, 3, 1, 2)

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