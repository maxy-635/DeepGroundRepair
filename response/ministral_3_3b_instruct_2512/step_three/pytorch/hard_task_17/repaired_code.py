import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Global Average Pooling and Fully Connected Layers
        self.block1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(32, 32),  # Assuming input channels is 32
            nn.Linear(32, 32)
        )

        # Extract weights from Block 1
        self.weights = nn.Parameter(torch.randn(32, 32, 1, 1), requires_grad=True)

        # Block 2: Convolutional Layers and Max Pooling
        self.block2_conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.block2_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.block2_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Branch from Block 1 to Block 2
        self.branch = nn.Sequential(
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU()
        )

        # Fully Connected Layers for Classification
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)  # Assuming CIFAR-10 has 10 classes

    def forward(self, x):
        # the initial shape of x is: (1, 32, 32, 3)
        # Block 1 Forward Pass
        block1_output = self.block1(x)  # output tensor shape: (1, 32)
        # Reshape weights to match input shape
        weights = self.weights.view(32, 32, 1, 1)  # output tensor shape: (32, 32, 1, 1)
        weighted_features = torch.mul(x, weights)  # output tensor shape: (32, 32, 32, 3)
        block1_output = torch.sum(weighted_features, dim=1)  # output tensor shape: (32, 32, 3)

        # Block 2 Forward Pass
        block2_output = self.block2_conv1(x)  # output tensor shape: (1, 64, 32, 3)
        block2_output = F.relu(block2_output)  # output tensor shape: (1, 64, 32, 3)
        block2_output = self.block2_conv2(block2_output)  # output tensor shape: (1, 64, 32, 3)
        block2_output = F.relu(block2_output)  # output tensor shape: (1, 64, 32, 3)
        block2_output = self.block2_pool(block2_output)  # output tensor shape: (1, 64, 16, 1)

        # Branch Forward Pass
        branch_output = self.branch(block1_output)
        branch_output = branch_output.view(-1, 64, 1, 1)
        branch_output = torch.mul(branch_output, x)

        # Fuse outputs
        fused_output = block2_output + branch_output

        # Fully Connected Layers
        fused_output = fused_output.view(fused_output.size(0), -1)
        fused_output = F.relu(self.fc1(fused_output))
        output = self.fc2(fused_output)

        return output

if __name__ == "__main__":
    # Dummy input tensor for CIFAR-10 (32x32x3)
    input_tensor = torch.randn(1, 32, 32, 3)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)