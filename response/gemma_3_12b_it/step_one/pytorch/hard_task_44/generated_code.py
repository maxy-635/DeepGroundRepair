import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.block1_conv2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.block1_conv3 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.block1_dropout = nn.Dropout(p=0.25)
        self.block1_concat = lambda x: torch.cat(x, dim=1)

        # Block 2
        self.block2_branch1_conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.block2_branch2_conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.block2_branch2_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.block2_branch3_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block2_branch3_conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.block2_concat = lambda x: torch.cat(x, dim=1)

        # Output
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=128 * 8 * 8, out_features=10)  # Assuming input size of 32x32 after convs and pooling

    def forward(self, x):
        # Block 1
        x1 = self.block1_conv1(x)
        x2 = self.block1_conv2(x)
        x3 = self.block1_conv3(x)
        x_block1 = self.block1_dropout(torch.cat([x1, x2, x3], dim=1))

        # Block 2
        x_branch1 = self.block2_branch1_conv1(x_block1)
        x_branch2 = torch.cat([self.block2_branch2_conv1(x_block1), self.block2_branch2_conv2(x_block1)], dim=1)
        x_branch3 = self.block2_branch3_pool(x_block1)
        x_branch3 = self.block2_branch3_conv1(x_branch3)

        x_block2 = self.block2_concat([x_branch1, x_branch2, x_branch3])

        # Output
        x = self.flatten(x_block2)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)