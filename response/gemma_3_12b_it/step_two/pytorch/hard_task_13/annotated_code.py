import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: Parallel branches for feature extraction
        self.conv1x1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool3x3 = nn.MaxPool2d(kernel_size='mask', stride='mask', padding='mask')

        # Concatenate the outputs of the parallel branches
        # Second block: Dimensionality reduction and feature weighting
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.fc_output = nn.Linear(in_features=64, out_features=10)  # Assuming 10 classes in CIFAR-10

    def forward(self, x):
        # the initial shape of x is: (32, 3, 32, 32)
        # First block
        branch1 = self.conv1x1(x)  # output tensor shape: (32, 64, 32, 32)
        branch2 = self.conv3x3_1(x)  # output tensor shape: (32, 64, 32, 32)
        branch3 = self.conv5x5(x)  # output tensor shape: (32, 64, 32, 32)
        branch4 = self.pool3x3(x)  # output tensor shape of this layer should be (32, 3, 32, 32)

        # Concatenate the branches
        merged_features = torch.cat((branch1, branch2, branch3, branch4), dim=1)

        # Second block
        pooled_features = self.global_avg_pool(merged_features)
        x = torch.reshape(pooled_features, (pooled_features.size(0), -1))
        x = self.fc1(x)
        x = self.fc2(x)

        # Reshape and multiply with the input feature map
        weights = x
        weights = torch.reshape(weights, (weights.size(0), weights.size(1), 1, 1))
        weighted_features = merged_features * weights
        
        # Final fully connected layer for classification
        x = torch.reshape(weighted_features, (weighted_features.size(0), -1))
        x = self.fc_output(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (batch_size, channels, height, width)
    input_tensor = torch.randn(32, 3, 32, 32)  # Example: batch size 32, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output tensor shape:", output_tensor.shape)