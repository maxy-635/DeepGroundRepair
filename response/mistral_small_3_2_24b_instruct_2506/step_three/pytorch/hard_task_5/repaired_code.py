import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Split into three groups and process with 1x1 convolutions
        self.block1_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.block1_conv2 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.block1_conv3 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Block 2: Channel shuffling
        self.block2_permute = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=96, out_channels=96, kernel_size=1, stride=1, padding=0),
                nn.Permutation((0, 2, 3, 1))
            )
        ])

        # Block 3: 3x3 depthwise separable convolution
        self.block3_depthwise = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, groups=96)
        self.block3_pointwise = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=1, stride=1, padding=0)

        # Branch connection
        self.branch_conv = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=1, stride=1, padding=0)

        # Final fully connected layer
        self.fc = nn.Linear(in_features=96 * 32 * 32, out_features=10)

    def forward(self, x):
        # Block 1: Split and process
        x1 = self.block1_conv1(x)
        x2 = self.block1_conv2(x)
        x3 = self.block1_conv3(x)
        x = torch.cat((x1, x2, x3), dim=1)

        # Block 2: Channel shuffling
        x = self.block2_permute[0](x)
        x = torch.reshape(x, (x.size(0), 96, 32, 32))

        # Block 3: Depthwise separable convolution
        x = self.block3_depthwise(x)
        x = self.block3_pointwise(x)

        # Branch connection
        branch = self.branch_conv(x)
        x = x + branch

        # Final classification
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)