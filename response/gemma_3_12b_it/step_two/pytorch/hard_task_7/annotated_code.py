import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.group1_conv1x1 = nn.Conv2d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')
        self.group1_depthwise = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32)
        self.group1_conv1x1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)

        self.block2_group_num = 4
        self.block2_linear = nn.Linear(in_features=32 * 28 * 28, out_features=10)  # Assuming 28x28 image size after conv layers

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Initial Convolution
        x = self.initial_conv(x)  # output tensor shape: (1, 32, 28, 28)

        # Block 1
        group1_size = x.shape[1] // 2  # output tensor shape: (1)
        x1 = x[:, :group1_size, :, :]  # output tensor shape: (1, 16, 28, 28)
        x2 = x[:, group1_size:, :, :]  # output tensor shape: (1, 16, 28, 28)

        x1 = self.group1_conv1x1(x1)
        x1 = self.group1_depthwise(x1)
        x1 = self.group1_conv1x1_2(x1)

        x = torch.cat((x1, x2), dim=1)

        # Block 2
        shape = x.shape
        height, width, channels = shape[2], shape[3], shape[1]
        groups = self.block2_group_num
        channels_per_group = channels // groups

        x = x.reshape(shape[0], height, width, groups, channels_per_group)
        x = x.permute(0, 1, 3, 2, 4)
        x = x.reshape(shape[0], height, width, channels)
        
        # Flatten and Linear Layer
        x = x.flatten(start_dim=1)
        x = self.block2_linear(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Dummy input: Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)