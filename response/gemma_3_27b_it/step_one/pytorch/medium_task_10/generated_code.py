import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Level 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.relu1 = nn.ReLU(inplace=True)

        # Basic Block
        self.basic_block = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True)
        )

        # Level 2
        self.conv2_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(num_features=16)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(num_features=16)
        self.relu2_2 = nn.ReLU(inplace=True)
        
        # Level 3
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=16)
        self.relu3 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=8)
        self.fc = nn.Linear(in_features=16 * 1 * 1, out_features=10)

    def forward(self, x):
        # Level 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Basic Block
        residual = x
        x = self.basic_block(x)
        x = x + residual

        # Level 2
        residual = x
        branch = self.conv2_1(x)
        branch = self.bn2_1(branch)
        branch = self.relu2_1(branch)
        
        x = self.basic_block(x)
        x = x + branch

        branch2 = self.conv2_2(x)
        branch2 = self.bn2_2(branch2)
        branch2 = self.relu2_2(branch2)
        x = self.basic_block(x)
        x = x + branch2

        # Level 3
        residual = self.conv3(x)
        residual = self.bn3(residual)
        residual = self.relu3(residual)
        x = x + residual

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)