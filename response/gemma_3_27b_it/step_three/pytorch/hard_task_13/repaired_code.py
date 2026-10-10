import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Parallel branches for feature extraction
        self.conv1x1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)

        # Block 2: Dimensionality reduction and feature weighting
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(in_features=32, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=32)
        self.fc_output = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1: Feature extraction
        branch1 = self.conv1x1(x)  # output tensor shape: (1, 32, 32, 32)
        branch2 = self.conv3x3(x)  # output tensor shape: (1, 32, 32, 32)
        branch3 = self.conv5x5(x)  # output tensor shape: (1, 32, 32, 32)
        branch4 = self.maxpool(x)  # output tensor shape of this layer should be (1, 3, 32, 32)

        # Concatenate features
        features = torch.cat((branch1, branch2, branch3, branch4), dim=1)

        # Block 2: Dimensionality reduction and feature weighting
        pooled_features = self.global_avg_pool(features)
        pooled_features = torch.flatten(pooled_features, 1)

        weights = self.fc1(pooled_features)
        weights = self.fc2(weights)
        weights = torch.reshape(weights, (features.shape[0], features.shape[1], 1, 1))
        weighted_features = torch.mul(features, weights)
        
        # Final fully connected layer
        output = self.fc_output(weighted_features.view(features.shape[0], -1))
        output = F.softmax(output, dim=1)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)