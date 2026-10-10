import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path components
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)

        # Branch path components
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Final classification layers
        self.fc3 = nn.Linear(in_features=64, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=64)
        self.fc5 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # Main path
        pooled = self.global_avg_pool(x)
        pooled = torch.flatten(pooled, 1)
        weights = self.fc1(pooled)
        weights = F.relu(weights)
        weights = self.fc2(weights)
        weights = torch.sigmoid(weights)  # Ensure weights are between 0 and 1
        weights = weights.view(-1, 64, 1, 1)  # Reshape to match input channels
        main_output = x * weights

        # Branch path
        branch_output = self.conv(x)

        # Combine paths
        combined = main_output + branch_output

        # Final classification
        combined = torch.flatten(combined, 1)
        combined = F.relu(self.fc3(combined))
        combined = F.relu(self.fc4(combined))
        output = self.fc5(combined)
        output = F.softmax(output, dim=1)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 64, 32, 32)  # Example input shape for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)