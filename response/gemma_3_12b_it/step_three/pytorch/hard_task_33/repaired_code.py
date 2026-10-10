import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.depthwise = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels)
        self.conv2 = nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        identity = x  # output tensor shape: (1, 1, 28, 28)
        x = self.conv1(x)  # output tensor shape: (1, 32, 28, 28)
        x = F.relu(x)  # output tensor shape: (1, 32, 28, 28)
        x = self.depthwise(x)  # output tensor shape: (1, 32, 28, 28)
        x = F.relu(x)  # output tensor shape: (1, 32, 28, 28)
        x = self.conv2(x)  # output tensor shape: (1, 1, 28, 28)
        x = x + identity  # output tensor shape: (1, 1, 28, 28)
        return x

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.branch1 = Block(1, 32)
        self.branch2 = Block(1, 32)
        self.branch3 = Block(1, 32)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=2352, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        out1 = self.branch1(x)  # output tensor shape: (1, 1, 28, 28)
        out2 = self.branch2(x)  # output tensor shape: (1, 1, 28, 28)
        out3 = self.branch3(x)  # output tensor shape: (1, 1, 28, 28)
        output = torch.cat((out1, out2, out3), dim=1)  # output tensor shape: (1, 3, 28, 28)
        output = self.flatten(output)  # output tensor shape: (1, 2352)
        output = self.linear(output)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)