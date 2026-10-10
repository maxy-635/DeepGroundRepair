import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, 64)  # Assuming input channels are 64 (common for CIFAR-10)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(in_features='mask', out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 64, 32, 32)
        # Global average pooling
        x = self.global_avg_pool(x)  # output tensor shape: (1, 64, 1, 1)
        x = x.view(x.size(0), -1)  # output tensor shape: (1, 64)

        # Two fully connected layers to generate weights
        x = self.fc1(x)  # output tensor shape: (1, 64)
        x = self.fc2(x)  # output tensor shape: (1, 64)

        # Reshape weights to match input channels and multiply element-wise
        weights = x.view(-1, 64, 1, 1)  # output tensor shape: (1, 64, 1, 1)
        x = torch.mul(x, weights)  # output tensor shape: (1, 64, 1, 64)

        # Flatten and final fully connected layer
        x = torch.flatten(x, 1)  # output tensor shape: (1, 4096)
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 64, 32, 32)  # (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor)