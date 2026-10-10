import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path components
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Fully connected layers to generate weights
        self.fc1 = nn.Linear(in_features=32*32*3, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=32*32*3)  # Assuming input channels=3 for CIFAR-10

        # Branch path components
        self.conv_branch = nn.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Final classification layers
        self.fc3 = nn.Linear(in_features=32*32*3, out_features=512)
        self.fc4 = nn.Linear(in_features=512, out_features=256)
        self.fc5 = nn.Linear(in_features=256, out_features=10)  # CIFAR-10 has 10 classes

        # Activation functions
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Main path
        pooled = self.global_avg_pool(x)
        pooled = pooled.view(pooled.size(0), -1)  # Flatten
        weights = self.fc1(pooled)
        weights = self.relu(weights)
        weights = self.fc2(weights)
        weights = self.sigmoid(weights)  # Using sigmoid to get weights between 0 and 1
        weights = weights.view(-1, 3, 32, 32)  # Reshape to match input dimensions

        # Element-wise multiplication with original input
        weighted_features = x * weights

        # Branch path
        branch_features = self.conv_branch(x)

        # Combine paths
        combined = weighted_features + branch_features

        # Flatten for classification
        combined = combined.view(combined.size(0), -1)

        # Final classification layers
        out = self.fc3(combined)
        out = self.relu(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)