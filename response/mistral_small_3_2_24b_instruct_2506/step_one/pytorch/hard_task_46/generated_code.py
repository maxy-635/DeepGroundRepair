import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Split into three groups with different kernel sizes
        self.conv1x1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)

        # Second block: Multiple branches for enhanced feature extraction
        # Branch 1: 3x3 convolution
        self.branch1_conv = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Branch 2: 1x1 convolution followed by two 3x3 convolutions
        self.branch2_conv1 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch2_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch2_conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Branch 3: Max pooling
        self.branch3_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # Global average pooling and fully connected layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(in_features=192, out_features=10)

        # Activation function
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # First block: Split into three groups and concatenate
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.conv5x5(x)
        x = torch.cat([x1, x2, x3], dim=1)

        # Second block: Multiple branches
        # Branch 1
        branch1 = self.branch1_conv(x)

        # Branch 2
        branch2 = self.branch2_conv1(x)
        branch2 = self.relu(branch2)
        branch2 = self.branch2_conv2(branch2)
        branch2 = self.relu(branch2)
        branch2 = self.branch2_conv3(branch2)

        # Branch 3
        branch3 = self.branch3_pool(x)

        # Concatenate all branches
        x = torch.cat([branch1, branch2, branch3], dim=1)

        # Global average pooling and fully connected layer
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Dummy input tensor for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])