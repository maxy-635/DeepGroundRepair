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
        self.fc3 = nn.Linear(in_features=65536, out_features=10)
        self.fc4 = nn.Linear(in_features=self.input_channels, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (32, 64, 32, 32)
        # Main path
        pooled = self.global_avg_pool(x)  # output tensor shape: (32, 64, 1, 1)
        pooled = torch.flatten(pooled, 1)  # output tensor shape: (32, 64)
        weights = self.fc1(pooled)  # output tensor shape: (32, 64)
        weights = self.fc2(weights)  # output tensor shape: (32, 64)
        weights = torch.reshape(weights, (-1, self.input_channels, 1, 1))  # output tensor shape: (32, 64, 1, 1)
        weighted_features = torch.mul(x, weights)  # output tensor shape: (32, 64, 32, 32)

        # Branch path
        branch_features = x  # output tensor shape: (32, 64, 32, 32)

        # Combine paths
        combined = torch.add(weighted_features, branch_features)  # output tensor shape: (32, 64, 32, 32)
        combined = torch.flatten(combined, 1)  # output tensor shape: (32, 65536)

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