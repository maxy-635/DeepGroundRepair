import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Assuming input features have 64 channels (common for CIFAR-10)
        self.input_channels = 64
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features=self.input_channels, out_features=self.input_channels)
        self.fc2 = nn.Linear(in_features=self.input_channels, out_features=self.input_channels)
        self.fc3 = nn.Linear(in_features=self.input_channels, out_features=10)  # CIFAR-10 has 10 classes
        self.fc4 = nn.Linear(in_features=self.input_channels, out_features=10)

    def forward(self, x):
        # Main path
        pooled = self.global_avg_pool(x)
        pooled = torch.flatten(pooled, 1)
        weights = self.fc1(pooled)
        weights = self.fc2(weights)
        weights = torch.reshape(weights, (-1, self.input_channels, 1, 1))
        weighted_features = torch.mul(x, weights)

        # Branch path
        branch_features = x

        # Combine paths
        combined = torch.add(weighted_features, branch_features)
        combined = torch.flatten(combined, 1)

        # Final classification layers
        out = self.fc3(combined)
        out = self.fc4(out)
        out = F.softmax(out, dim=1)

        return out

if __name__ == "__main__":
    model = DL_Model()
    # Assuming input shape is (batch_size, channels, height, width) for CIFAR-10
    input_tensor = torch.randn(32, 64, 32, 32)  # batch_size=32, channels=64, height=32, width=32
    output_tensor = model(input_tensor)
    print(output_tensor)