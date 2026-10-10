import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three parallel paths with different average pooling scales
        self.avg_pool1x1 = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.avg_pool2x2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.avg_pool4x4 = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)

        # Fully connected layer between Block 1 and Block 2
        self.fc1 = nn.Linear(in_features=1029, out_features=128)

        # Block 2: Three branches for feature extraction
        # Branch 1: 1x1 convolution followed by 3x3 convolution
        self.branch1_conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.branch1_conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)

        # Branch 2: 1x1 convolution followed by 1x7 convolution, 7x1 convolution, and 3x3 convolution
        self.branch2_conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.branch2_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 7), stride=1, padding=0)
        self.branch2_conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(7, 1), stride=1, padding=0)
        self.branch2_conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)

        # Branch 3: Average pooling
        self.branch3_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=0)

        # Final fully connected layers
        self.fc2 = nn.Linear(in_features=64 * 26 * 26, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Block 1: Three parallel paths
        x1 = self.avg_pool1x1(x)  # output tensor shape: (1, 1, 28, 28)
        x2 = self.avg_pool2x2(x)  # output tensor shape: (1, 1, 14, 14)
        x3 = self.avg_pool4x4(x)  # output tensor shape: (1, 1, 7, 7)

        # Flatten and concatenate
        x1_flat = torch.flatten(x1, start_dim=1)  # output tensor shape: (1, 784)
        x2_flat = torch.flatten(x2, start_dim=1)  # output tensor shape: (1, 196)
        x3_flat = torch.flatten(x3, start_dim=1)  # output tensor shape: (1, 49)
        x_combined = torch.cat((x1_flat, x2_flat, x3_flat), dim=1)  # output tensor shape: (1, 1029)

        # Fully connected layer
        x_fc = self.fc1(x_combined)

        # Reshape to 4D tensor for Block 2
        x_reshaped = torch.reshape(x_fc, (x_fc.size(0), 1, 8, 16))

        # Block 2: Three branches
        # Branch 1
        branch1 = self.branch1_conv1(x_reshaped)
        branch1 = self.branch1_conv2(branch1)

        # Branch 2
        branch2 = self.branch2_conv1(x_reshaped)
        branch2 = self.branch2_conv2(branch2)
        branch2 = self.branch2_conv3(branch2)
        branch2 = self.branch2_conv4(branch2)

        # Branch 3
        branch3 = self.branch3_pool(x_reshaped)

        # Concatenate branches
        x_block2 = torch.cat((branch1, branch2, branch3), dim=1)

        # Flatten and pass through final fully connected layers
        x_flat = torch.flatten(x_block2, start_dim=1)
        x_fc2 = self.fc2(x_flat)
        x_fc3 = self.fc3(x_fc2)

        return x_fc3

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Dummy input tensor for MNIST (batch_size=1, channels=1, height=28, width=28)
    output_tensor = model(input_tensor)
    print(output_tensor)