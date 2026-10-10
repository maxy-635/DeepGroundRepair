import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Split and depthwise separable convolutions
        self.block1_group1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(num_features=32)
        )
        self.block1_group2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(num_features=32)
        )
        self.block1_group3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.BatchNorm2d(num_features=32)
        )

        # Concatenate outputs from the three groups
        self.block1_concat = nn.Sequential(
            nn.Conv2d(in_channels=3 * 32, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(num_features=64)
        )

        # Second block: Multiple branches
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(num_features=64)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3)),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0)),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(num_features=64)
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(num_features=64)
        )
        self.block2_concat = nn.Sequential(
            nn.Conv2d(in_channels=3 * 64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(num_features=128)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=128, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # 10 classes for CIFAR-10

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First block
        x1 = self.block1_group1(x)  # output tensor shape: (1, 32, 32, 32)
        x2 = self.block1_group2(x)  # output tensor shape: (1, 32, 32, 32)
        x3 = self.block1_group3(x)  # output tensor shape: (1, 32, 32, 32)
        x = torch.cat((x1, x2, x3), dim=1)  # output tensor shape: (1, 96, 32, 32)
        x = self.block1_concat(x)  # output tensor shape: (1, 64, 32, 32)

        # Second block
        x1 = self.branch1(x)  # output tensor shape: (1, 64, 32, 32)
        x2 = self.branch2(x)  # output tensor shape: (1, 64, 32, 32)
        x3 = self.branch3(x)  # output tensor shape of this layer should be (1, 64, 32, 32)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.block2_concat(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)