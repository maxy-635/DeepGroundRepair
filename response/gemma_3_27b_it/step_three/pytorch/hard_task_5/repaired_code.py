import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_conv1 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.block1_conv2 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.block1_conv3 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)

        # Block 2
        self.block2_groups = 3

        # Block 3
        self.block3_depthwise = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0, groups=3)

        # Branch
        self.branch_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Fully connected layer
        self.fc = nn.Linear(in_features=3 * 32 * 32, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1
        group1 = self.block1_conv1(x)  # output tensor shape: (1, 1, 32, 32)
        group2 = self.block1_conv2(x)  # output tensor shape: (1, 1, 32, 32)
        group3 = self.block1_conv3(x)  # output tensor shape: (1, 1, 32, 32)
        features = torch.cat([group1, group2, group3], dim=1)  # output tensor shape: (1, 3, 32, 32)

        # Block 2
        shape = features.shape  # output tensor shape: (unknown)
        height, width, channels = shape[2], shape[3], shape[1]  # output tensor shape: (1); (1); (1)
        features = torch.reshape(features, (height, width, self.block2_groups, channels // self.block2_groups))  # output tensor shape: (32, 32, 3, 1)
        features = torch.permute(features, (0, 2, 3, 1))  # output tensor shape: (32, 3, 1, 32)
        features = torch.reshape(features, (height, width, channels))  # output tensor shape: (32, 32, 3)

        # Block 3
        features = self.block3_depthwise(features)
        features = F.relu(features)

        # Branch
        branch_output = self.branch_conv(x)

        # Addition
        output = torch.add(features, branch_output)
        output = F.relu(output)

        # Flatten and fully connected layer
        output = output.view(-1, 3 * 32 * 32)
        output = self.fc(output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)