import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_group1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=16)
        )
        self.block1_group2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16)
        )
        self.block1_group3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=16)
        )

        # Block 2
        self.block2_path1 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=1, stride=1, padding=0)
        )
        self.block2_path2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=1, stride=1, padding=0)
        )
        self.block2_path3 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size='mask', stride='mask', padding='mask')
        )
        self.block2_path4 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)
        )

        # Classifier
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=32*8*8, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1
        x1 = self.block1_group1(x)  # output tensor shape: (1, 16, 32, 32)
        x2 = self.block1_group2(x)  # output tensor shape: (1, 16, 32, 32)
        x3 = self.block1_group3(x)  # output tensor shape: (1, 16, 32, 32)
        x = torch.cat((x1, x2, x3), dim=1)  # output tensor shape: (1, 48, 32, 32)

        # Block 2
        x_path1 = self.block2_path1(x)  # output tensor shape: (1, 32, 32, 32)
        x_path2 = self.block2_path2(x)  # output tensor shape: (1, 32, 32, 32)
        x_path3_1 = self.block2_path3(x)  # output tensor shape of this layer should be (1, 32, 28, 28)
        x_path3_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)(x_path3_1)  # output tensor shape: (1, 32, 28, 28)
        x_path3 = torch.cat((x_path3_1, x_path3_2), dim=1)
        x_path4_1 = self.block2_path4(x)
        x_path4_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0)(x_path4_1)
        x_path4 = torch.cat((x_path4_1, x_path4_2), dim=1)

        x = torch.cat((x_path1, x_path2, x_path3, x_path4), dim=1)

        # Classifier
        x = self.flatten(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)