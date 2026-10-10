import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial 1x1 convolutional layer
        self.initial_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Branch 1: Local feature extraction
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

        # Branch 2: Average pooling and transposed convolution
        self.branch2_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.branch2_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch2_transpose = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)

        # Branch 3: Average pooling and transposed convolution
        self.branch3_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.branch3_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch3_transpose = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)

        # Concatenation and refinement
        self.concat_conv = nn.Conv2d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')

        # Fully connected layer for classification
        self.fc = nn.Linear(in_features=64 * 8 * 8, out_features=10)  # Assuming input size of 32x32 after initial conv and branches

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Initial convolution
        x = self.initial_conv(x)  # output tensor shape: (1, 64, 32, 32)

        # Branch 1
        branch1_out = self.branch1(x)  # output tensor shape: (1, 64, 32, 32)

        # Branch 2
        branch2_out = self.branch2_pool(x)  # output tensor shape: (1, 64, 16, 16)
        branch2_out = self.branch2_conv(branch2_out)  # output tensor shape: (1, 64, 16, 16)
        branch2_out = self.branch2_transpose(branch2_out)  # output tensor shape: (1, 64, 32, 32)

        # Branch 3
        branch3_out = self.branch3_pool(x)  # output tensor shape: (1, 64, 16, 16)
        branch3_out = self.branch3_conv(branch3_out)  # output tensor shape: (1, 64, 16, 16)
        branch3_out = self.branch3_transpose(branch3_out)  # output tensor shape: (1, 64, 32, 32)

        # Concatenation
        x = torch.cat((branch1_out, branch2_out, branch3_out), dim=1)  # output tensor shape: (1, 192, 32, 32)

        # Refinement
        x = self.concat_conv(x)

        # Flatten for fully connected layer
        x = x.view(x.size(0), -1)

        # Classification
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)