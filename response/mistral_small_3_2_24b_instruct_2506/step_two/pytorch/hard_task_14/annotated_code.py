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
        self.fc3 = nn.Linear(in_features='mask', out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=64)
        self.fc5 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 64, 32, 32)
        # Main path
        pooled = self.global_avg_pool(x)  # output tensor shape: (1, 64, 1, 1)
        pooled = torch.flatten(pooled, 1)  # output tensor shape: (1, 64)
        weights = self.fc1(pooled)  # output tensor shape: (1, 64)
        weights = F.relu(weights)  # output tensor shape: (1, 64)
        weights = self.fc2(weights)  # output tensor shape: (1, 64)
        weights = torch.sigmoid(weights)  # output tensor shape: (1, 64)
        weights = weights.view(-1, 64, 1, 1)  # output tensor shape: (1, 64, 1, 1)
        main_output = x * weights  # output tensor shape: (1, 64, 32, 32)

        # Branch path
        branch_output = self.conv(x)  # output tensor shape: (1, 64, 32, 32)

        # Combine paths
        combined = main_output + branch_output  # output tensor shape: (1, 64, 32, 32)

        # Final classification
        combined = torch.flatten(combined, 1)  # output tensor shape: (1, 65536)
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