import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three parallel average pooling paths
        self.avg_pool_1x1 = nn.AvgPool2d(kernel_size=1, stride=1)
        self.avg_pool_2x2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool_4x4 = nn.AvgPool2d(kernel_size=4, stride=4)

        # Flatten and concatenate the outputs
        self.flatten = nn.Flatten()

        # Fully connected layer between Block 1 and Block 2
        self.fc1 = nn.Linear(in_features=1029, out_features=128)

        # Reshape for Block 2 input
        self.reshape = nn.Sequential(
            nn.Linear(128, 16*16*8),
            nn.ReLU(),
            nn.Unflatten(1, (8, 16, 16))
        )

        # Block 2: Three branches for feature extraction
        # Branch 1: 1x1 conv + 3x3 conv
        self.branch1_conv1x1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, stride=1)
        self.branch1_conv3x3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Branch 2: 1x1 conv + 1x7 conv + 7x1 conv + 3x3 conv
        self.branch2_conv1x1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, stride=1)
        self.branch2_conv1x7 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.branch2_conv7x1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.branch2_conv3x3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Branch 3: Average pooling
        self.branch3_avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Final fully connected layers
        self.fc2 = nn.Linear(in_features=32*16*16 + 32*16*16 + 8*8*8, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Block 1 processing
        pool1 = self.avg_pool_1x1(x)  # output tensor shape: (1, 1, 28, 28)
        pool2 = self.avg_pool_2x2(x)  # output tensor shape: (1, 1, 14, 14)
        pool4 = self.avg_pool_4x4(x)  # output tensor shape: (1, 1, 7, 7)

        # Flatten and concatenate
        pool1_flat = self.flatten(pool1)  # output tensor shape: (1, 784)
        pool2_flat = self.flatten(pool2)  # output tensor shape: (1, 196)
        pool4_flat = self.flatten(pool4)  # output tensor shape: (1, 49)

        # Concatenate along the feature dimension
        combined = torch.cat((pool1_flat, pool2_flat, pool4_flat), dim=1)  # output tensor shape: (1, 1029)

        # Fully connected layer
        fc_out = self.fc1(combined)

        # Reshape for Block 2
        reshaped = self.reshape(fc_out)

        # Block 2 processing
        # Branch 1
        branch1 = self.branch1_conv1x1(reshaped)
        branch1 = F.relu(branch1)
        branch1 = self.branch1_conv3x3(branch1)
        branch1 = F.relu(branch1)

        # Branch 2
        branch2 = self.branch2_conv1x1(reshaped)
        branch2 = F.relu(branch2)
        branch2 = self.branch2_conv1x7(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.branch2_conv7x1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.branch2_conv3x3(branch2)
        branch2 = F.relu(branch2)

        # Branch 3
        branch3 = self.branch3_avg_pool(reshaped)

        # Flatten and concatenate branches
        branch1_flat = self.flatten(branch1)
        branch2_flat = self.flatten(branch2)
        branch3_flat = self.flatten(branch3)

        combined_branches = torch.cat((branch1_flat, branch2_flat, branch3_flat), dim=1)

        # Final classification
        output = self.fc2(combined_branches)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)