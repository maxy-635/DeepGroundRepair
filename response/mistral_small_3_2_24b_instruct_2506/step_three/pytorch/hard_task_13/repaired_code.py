import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: parallel branches
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        # Second block: dimensionality reduction and attention mechanism
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features=195, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.fc_final = nn.Linear(in_features=64, out_features=10)  # 10 classes for CIFAR-10

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First block: parallel branches
        branch1_out = self.branch1(x)  # output tensor shape: (1, 64, 32, 32)
        branch2_out = self.branch2(x)  # output tensor shape: (1, 64, 32, 32)
        branch3_out = self.branch3(x)  # output tensor shape: (1, 64, 32, 32)
        branch4_out = self.branch4(x)  # output tensor shape: (1, 3, 32, 32)

        # Concatenate the outputs of the parallel branches
        concatenated = torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], dim=1)  # output tensor shape: (1, 195, 32, 32)

        # Second block: attention mechanism
        pooled = self.global_avg_pool(concatenated)  # output tensor shape: (1, 195, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # output tensor shape: (1, 195)

        # Generate weights
        weights = self.fc1(pooled)
        weights = F.relu(weights)
        weights = self.fc2(weights)
        weights = torch.sigmoid(weights)
        weights = weights.view(-1, 64, 1, 1)

        # Apply attention
        attended = torch.mul(concatenated, weights)

        # Final classification
        attended = attended.view(attended.size(0), -1)
        output = self.fc_final(attended)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # Expected output shape: torch.Size([1, 10])