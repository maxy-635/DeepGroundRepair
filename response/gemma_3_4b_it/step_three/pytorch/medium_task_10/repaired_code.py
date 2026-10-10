import torch
import torch.nn as nn
import torch.optim as optim

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)

        # Residual block 1
        self.res_block1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # Residual block 2
        self.res_block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Global branch
        self.conv_global = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=3, stride=1, padding=1)

        # Average pooling
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1)

        # Fully connected layer
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Initial convolution and batch normalization
        x = self.conv1(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.bn1(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.relu1(x)  # output tensor shape: (1, 16, 32, 32)

        # Residual block 1
        x = self.res_block1(x)  # output tensor shape: (1, 16, 32, 32)

        # Residual block 2
        x = self.res_block2(x)  # output tensor shape: (1, 32, 32, 32)

        # Global branch
        x = self.conv_global(x)

        # Addition
        x = torch.add(x, self.res_block2(x))

        # Average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)