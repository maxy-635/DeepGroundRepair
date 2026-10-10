import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn_shortcut = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = self.shortcut(x)
        identity = self.bn_shortcut(identity)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = out + identity
        return out

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.initial_conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.block1 = BasicBlock(16, 16)
        self.block2 = BasicBlock(16, 16)
        self.branch_conv = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(16 * 5 * 5, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.block1(x)
        branch = self.branch_conv(x)
        x = self.block2(x)
        x = x + branch
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)