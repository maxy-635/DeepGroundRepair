import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Weight generation and multiplication
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_features='mask', out_features=3 * 32 * 32)
        self.fc2 = nn.Linear(3 * 32 * 32, 3 * 32 * 32)
        self.relu = nn.ReLU(inplace=True)

        # Block 2: Feature extraction
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Final classification layers
        self.fc3 = nn.Linear(64 * 8 * 8, 512)  # After maxpool, dimensions are halved twice
        self.fc4 = nn.Linear(512, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1: Weight generation
        weights = self.global_avg_pool(x)  # output tensor shape: (1, 3, 1, 1)
        weights = weights.view(weights.size(0), -1)  # output tensor shape: (1, 3)
        weights = self.fc1(weights)
        weights = self.relu(weights)
        weights = self.fc2(weights)
        weights = self.relu(weights)
        weights = weights.view(weights.size(0), 3, 32, 32)

        # Weighted feature output
        weighted_features = x * weights

        # Block 2: Feature extraction
        features = self.relu(self.conv1(x))
        features = self.relu(self.conv2(features))
        features = self.maxpool(features)

        # Branch connection
        branch_features = weighted_features
        branch_features = self.maxpool(branch_features)
        branch_features = self.relu(self.conv1(branch_features))
        branch_features = self.relu(self.conv2(branch_features))
        branch_features = self.maxpool(branch_features)

        # Fusion
        fused_features = features + branch_features

        # Classification
        fused_features = fused_features.view(fused_features.size(0), -1)
        output = self.relu(self.fc3(fused_features))
        output = self.fc4(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)