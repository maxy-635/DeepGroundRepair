import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Assuming input has 3 channels
        in_channels = 3
        num_kernels = in_channels // 3

        # Split into three groups, apply 1x1 conv, and average pool
        self.group1_conv = nn.Conv2d(in_channels=1, out_channels=num_kernels, kernel_size=1, stride=1, padding=0)
        self.group2_conv = nn.Conv2d(in_channels=1, out_channels=num_kernels, kernel_size=1, stride=1, padding=0)
        self.group3_conv = nn.Conv2d(in_channels=1, out_channels=num_kernels, kernel_size=1, stride=1, padding=0)

        self.group1_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.group2_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.group3_pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # Concatenate the results
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=num_kernels * 3 * 16 * 16, out_features=128)  # Adjusted input size based on CIFAR-10 and pooling
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Split input into three groups along the channel dimension
        group1 = x[:, :1, :, :]
        group2 = x[:, 1:2, :, :]
        group3 = x[:, 2:, :, :]

        # Apply 1x1 convolutions to each group
        group1 = self.group1_conv(group1)
        group2 = self.group2_conv(group2)
        group3 = self.group3_conv(group3)

        # Apply average pooling to each group
        group1 = self.group1_pool(group1)
        group2 = self.group2_pool(group2)
        group3 = self.group3_pool(group3)

        # Concatenate the groups along the channel dimension
        x = torch.cat((group1, group2, group3), dim=1)

        # Flatten and pass through fully connected layers
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)