import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, 64)  # Assuming input channels are 64 (common for CIFAR-10)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64 * 32 * 32, 10)  # Assuming input size is 32x32 (CIFAR-10)

    def forward(self, x):
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, channels)

        # Two fully connected layers to generate weights
        x = self.fc1(x)
        x = self.fc2(x)

        # Reshape weights to match input channels and multiply element-wise
        weights = x.view(-1, 64, 1, 1)  # Assuming 64 channels
        x = torch.mul(x, weights)

        # Flatten and final fully connected layer
        x = torch.flatten(x, 1)
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 64, 32, 32)  # (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor)