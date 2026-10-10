import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        # the initial shape of x is: (1, 16, 32, 32)
        residual = x  # output tensor shape: (1, 16, 32, 32)
        x = self.conv1(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.bn1(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.relu1(x)  # output tensor shape: (1, 16, 32, 32)
        return x + residual

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.conv_initial = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn_initial = nn.BatchNorm2d(num_features=16)
        self.relu_initial = nn.ReLU(inplace=True)

        # First Basic Block
        self.block1 = BasicBlock(in_channels=16, out_channels=16)

        # Second Basic Block
        self.block2 = BasicBlock(in_channels=16, out_channels=16)

        # Branch convolutional layer
        self.conv_branch = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn_branch = nn.BatchNorm2d(num_features=16)
        self.relu_branch = nn.ReLU(inplace=True)

        # Average pooling layer
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=8)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layer
        self.fc = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x = self.conv_initial(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.bn_initial(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.relu_initial(x)  # output tensor shape: (1, 16, 32, 32)

        x = self.block1(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.block2(x)  # output tensor shape: (1, 16, 32, 32)

        branch = self.conv_branch(x)  # output tensor shape: (1, 16, 32, 32)
        branch = self.bn_branch(branch)  # output tensor shape: (1, 16, 32, 32)
        branch = self.relu_branch(branch)  # output tensor shape: (1, 16, 32, 32)

        x = x + branch  # output tensor shape: (1, 16, 32, 32)

        x = self.avgpool(x)  # output tensor shape: (1, 16, 4, 4)
        x = self.flatten(x)  # output tensor shape: (1, 256)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)