import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_primary_conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.block1_primary_depthwise = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, groups=16)
        self.block1_primary_conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0)

        self.block1_branch_depthwise = nn.Conv2d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask', groups='mask')
        self.block1_branch_conv = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0)

        # Block 2
        self.block2_linear = nn.Linear(16 * 16 * 28 * 28, 10)  # Assuming input size is 28x28

    def forward(self, x):
        # Block 1
        primary_path = F.relu(self.block1_primary_conv1(x))
        primary_path = F.relu(self.block1_primary_depthwise(primary_path))
        primary_path = F.relu(self.block1_primary_conv2(primary_path))

        branch_path = F.relu(self.block1_branch_depthwise(x))
        branch_path = F.relu(self.block1_branch_conv(branch_path))

        x = torch.cat((primary_path, branch_path), dim=1)

        # Block 2
        shape = x.shape
        x = torch.reshape(x, (shape[0], shape[1], shape[2], shape[3]))

        groups = 4
        channels_per_group = x.shape[1] // groups
        x = torch.reshape(x, (shape[0], shape[2], shape[3], groups, channels_per_group))
        x = torch.permute(x, (0, 2, 3, 4, 1))
        x = torch.reshape(x, (shape[0], shape[2], shape[3], shape[1]))

        # Fully connected layer
        x = x.view(x.size(0), -1)
        x = self.block2_linear(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)