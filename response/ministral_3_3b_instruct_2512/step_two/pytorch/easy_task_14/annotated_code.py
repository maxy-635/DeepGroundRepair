import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Global Average Pooling Layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Fully Connected Layers for generating weights
        self.fc1 = nn.Linear(in_features='mask', out_features=1024)
        self.fc2 = nn.Linear(1024, 1024)

        # Reshape and multiply weights with input
        self.weights = nn.Parameter(torch.randn(1024, 1024))

        # Flatten and final classification layer
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(1024, 10)  # Assuming 10 output classes for CIFAR-10

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Apply global average pooling
        x = self.global_avg_pool(x)  # output tensor shape: (1, 3, 1, 1)

        # Generate weights using fully connected layers
        weights = self.fc1(x)
        weights = F.relu(weights)
        weights = self.fc2(weights)
        weights = F.relu(weights)

        # Reshape weights to match input dimensions
        weights = weights.view(weights.size(0), -1)
        weights = weights.view(weights.size(0), 1024, 1, 1)

        # Element-wise multiplication with input
        x = x * weights

        # Flatten and pass through final classification layer
        x = self.flatten(x)
        x = self.classifier(x)

        return x

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    # Assuming CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)