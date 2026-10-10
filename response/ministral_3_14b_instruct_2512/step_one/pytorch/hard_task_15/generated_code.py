import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Two fully connected layers to generate weights
        self.fc1 = nn.Linear(3 * 32 * 32, 1024)  # Assuming input channels=3, height=32, width=32
        self.fc2 = nn.Linear(1024, 3 * 32 * 32)  # Output channels same as input

        # Branch path (direct connection)
        self.branch = nn.Identity()

        # Final fully connected layers
        self.final_fc1 = nn.Linear(3 * 32 * 32, 1024)
        self.final_fc2 = nn.Linear(1024, 10)  # Assuming 10 classes for CIFAR-10

    def forward(self, x):
        # Main path
        pooled_features = self.global_avg_pool(x)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)  # Flatten
        weights = self.fc1(pooled_features)
        weights = self.fc2(weights)
        weights = weights.view(x.size(0), x.size(1), x.size(2), x.size(3))  # Reshape to match input shape
        weighted_features = x * weights

        # Branch path
        branch_features = self.branch(x)

        # Combine paths
        combined_features = weighted_features + branch_features
        combined_features = combined_features.view(combined_features.size(0), -1)  # Flatten

        # Final layers
        output = self.final_fc1(combined_features)
        output = self.final_fc2(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)