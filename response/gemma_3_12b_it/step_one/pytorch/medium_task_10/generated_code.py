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
        identity = self.shortcut(x)
        identity = self.bn_shortcut(identity)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = torch.add(out, identity)
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
        self.fc = nn.Linear(16 * 8 * 8, 10)  # Assuming CIFAR-10 input size is 32x32 and after pooling it becomes 8x8

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.block1(x)

        x = self.block2_1(x)
        branch_out = self.conv2_branch(x)
        branch_out = self.bn2_branch(branch_out)
        x = torch.add(x, branch_out)
        x = self.block2_2(x)

        x = self.conv3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)