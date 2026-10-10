import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups

        # Reshape to group channels
        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        # Permute dimensions to shuffle channels
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)

        return x

class Block1(nn.Module):
    def __init__(self, in_channels):
        super(Block1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')
        self.conv2 = nn.Conv2d(in_channels, in_channels//3, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, in_channels//3, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Split into three groups
        x1 = self.conv1(x[:, :x.size(1)//3, :, :])
        x2 = self.conv2(x[:, x.size(1)//3:2*x.size(1)//3, :, :])
        x3 = self.conv3(x[:, 2*x.size(1)//3:, :, :])

        # Concatenate along channel dimension
        out = torch.cat([x1, x2, x3], dim=1)
        return self.relu(out)

class Block2(nn.Module):
    def __init__(self, in_channels):
        super(Block2, self).__init__()
        self.shuffle = ChannelShuffle(groups=3)

    def forward(self, x):
        return self.shuffle(x)

class Block3(nn.Module):
    def __init__(self, in_channels):
        super(Block3, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.depthwise_conv(x)

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.block1 = Block1(in_channels=3)
        self.block2 = Block2(in_channels=3)
        self.block3 = Block3(in_channels=3)

        # Main path
        self.main_path = nn.Sequential(
            Block1(in_channels=3),
            Block2(in_channels=3),
            Block3(in_channels=3),
            Block1(in_channels=3)
        )

        # Branch path
        self.branch = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Final layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path
        main_out = self.main_path(x)

        # Branch path
        branch_out = self.branch(x)

        # Combine outputs
        combined = main_out + branch_out

        # Final classification
        out = self.avg_pool(combined)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)