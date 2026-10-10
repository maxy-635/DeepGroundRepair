import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn_shortcut = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # the initial shape of x is: (1, 16, 32, 32)
        identity = self.shortcut(x)  # output tensor shape: (1, 16, 32, 32)
        identity = self.bn_shortcut(identity)  # output tensor shape: (1, 16, 32, 32)
        out = self.conv1(x)  # output tensor shape: (1, 16, 32, 32)
        out = self.bn1(out)  # output tensor shape: (1, 16, 32, 32)
        out = self.relu(out)  # output tensor shape: (1, 16, 32, 32)
        out = torch.add(out, identity)  # output tensor shape: (1, 16, 32, 32)
        return out

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Level 1
        self.block1 = BasicBlock(16, 16)

        # Level 2
        self.block2_1 = BasicBlock(16, 16)
        self.block2_2 = BasicBlock(16, 16)
        self.conv2_branch = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn2_branch = nn.BatchNorm2d(16)

        # Level 3
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(in_features=4096, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x = self.conv1(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.bn1(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.relu(x)  # output tensor shape: (1, 16, 32, 32)

        x = self.block1(x)  # output tensor shape: (1, 16, 32, 32)

        x = self.block2_1(x)  # output tensor shape: (1, 16, 32, 32)
        branch_out = self.conv2_branch(x)  # output tensor shape: (1, 16, 32, 32)
        branch_out = self.bn2_branch(branch_out)  # output tensor shape: (1, 16, 32, 32)
        x = torch.add(x, branch_out)  # output tensor shape: (1, 16, 32, 32)
        x = self.block2_2(x)  # output tensor shape: (1, 16, 32, 32)

        x = self.conv3(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.avgpool(x)  # output tensor shape: (1, 16, 16, 16)
        x = torch.flatten(x, 1)  # output tensor shape: (1, 4096)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)