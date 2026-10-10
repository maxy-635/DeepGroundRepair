import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Parallel Average Pooling
        self.pool1x1 = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.pool2x2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.pool4x4 = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        self.flatten = nn.Flatten()

        # Fully connected layer between Block 1 and Block 2
        self.fc1 = nn.Linear(in_features=16*16*3, out_features=64)  # Assuming input image size is 32x32
        # Block 2: Feature Extraction Branches
        # Branch 1: 1x1 and 3x3 convolutions
        self.conv1x1_branch1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv3x3_branch1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Branch 2: 1x1, 1x7, 7x1, and 3x3 convolutions
        self.conv1x1_branch2 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv1x7_branch2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.conv7x1_branch2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(7, 1), stride=1, padding=(3, 0))
        self.conv3x3_branch2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Branch 3: Average Pooling
        self.avgpool_branch3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # Final Classification Layers
        self.fc2 = nn.Linear(in_features=64*8*8, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Block 1
        pool1x1_out = self.pool1x1(x)
        pool2x2_out = self.pool2x2(x)
        pool4x4_out = self.pool4x4(x)

        pool1x1_flat = self.flatten(pool1x1_out)
        pool2x2_flat = self.flatten(pool2x2_out)
        pool4x4_flat = self.flatten(pool4x4_out)

        block1_out = torch.concat([pool1x1_flat, pool2x2_flat, pool4x4_flat], dim=1)

        # FC Layer and Reshape
        block1_out = self.fc1(block1_out)
        block1_out = torch.reshape(block1_out, (x.size(0), 1, 8, 8))

        # Block 2
        # Branch 1
        branch1_out = F.relu(self.conv1x1_branch1(block1_out))
        branch1_out = F.relu(self.conv3x3_branch1(branch1_out))

        # Branch 2
        branch2_out = F.relu(self.conv1x1_branch2(block1_out))
        branch2_out = F.relu(self.conv1x7_branch2(branch2_out))
        branch2_out = F.relu(self.conv7x1_branch2(branch2_out))
        branch2_out = F.relu(self.conv3x3_branch2(branch2_out))
        
        # Branch 3
        branch3_out = self.avgpool_branch3(block1_out)
        
        # Concatenate Branches
        block2_out = torch.concat([branch1_out, branch2_out, branch3_out], dim=1)
        
        # Classification Layers
        block2_flat = self.flatten(block2_out)
        out = F.relu(self.fc2(block2_flat))
        out = self.fc3(out)

        return out


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 32, 32)  # Batch size 1, 1 channel, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)