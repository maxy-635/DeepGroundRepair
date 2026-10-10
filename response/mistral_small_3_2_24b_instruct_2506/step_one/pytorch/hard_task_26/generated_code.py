import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.initial_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Branch 1
        self.branch1_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Branch 2
        self.branch2_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.branch2_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch2_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Branch 3
        self.branch3_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.branch3_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Concatenation and final convolution
        self.final_conv = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Branch path
        self.branch_path_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Main path
        x_main = self.initial_conv(x)

        # Branch 1
        branch1 = self.branch1_conv(x_main)

        # Branch 2
        branch2 = self.branch2_pool(x_main)
        branch2 = self.branch2_conv(branch2)
        branch2 = self.branch2_upsample(branch2)

        # Branch 3
        branch3 = self.branch3_pool(x_main)
        branch3 = self.branch3_conv(branch3)
        branch3 = self.branch3_upsample(branch3)

        # Concatenate branches
        x_main = torch.cat([branch1, branch2, branch3], dim=1)
        x_main = self.final_conv(x_main)

        # Branch path
        x_branch = self.branch_path_conv(x)

        # Add main path and branch path
        x = x_main + x_branch

        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # CIFAR-10 input shape: (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor)