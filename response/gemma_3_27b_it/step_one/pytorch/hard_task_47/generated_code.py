import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Depthwise Separable Convolutions
        self.block1_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=32),
            F.relu
        )
        self.block1_3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            F.relu
        )
        self.block1_5 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=32),
            F.relu
        )

        # Block 2: Multiple Branches
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=32*3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            F.relu
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=32*3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            F.relu,
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            F.relu
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        )

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=64*3, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Block 1
        x1 = self.block1_1(x)
        x3 = self.block1_3(x)
        x5 = self.block1_5(x)
        x_block1 = torch.cat([x1, x3, x5], dim=1)

        # Block 2
        branch1_out = self.branch1(x_block1)
        branch2_out = self.branch2(x_block1)
        branch3_out = self.branch3(x_block1)

        x_block2 = torch.cat([branch1_out, branch2_out, branch3_out], dim=1)

        # Flatten
        x = x_block2.view(x.size(0), -1)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor (batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)