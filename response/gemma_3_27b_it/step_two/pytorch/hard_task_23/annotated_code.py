import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial 1x1 convolution
        self.initial_conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)

        # Branch 1: Local Feature Extraction
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        )

        # Branch 2: Downsampling and Upsampling
        self.branch2_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.branch2_conv = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch2_transpose_conv = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0)

        # Branch 3: Downsampling and Upsampling
        self.branch3_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.branch3_conv = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch3_transpose_conv = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0)

        # Concatenation and Refinement
        self.concat_conv = nn.Conv2d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')

        # Fully Connected Layer
        self.fc = nn.Linear(in_features=64 * 32 * 32, out_features=10)  # CIFAR-10 images are 32x32

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Initial Convolution
        x = self.initial_conv(x)  # output tensor shape: (1, 16, 32, 32)

        # Branch 1
        branch1_output = self.branch1(x)  # output tensor shape: (1, 32, 32, 32)

        # Branch 2
        branch2_output = self.branch2_pool(x)  # output tensor shape: (1, 16, 16, 16)
        branch2_output = self.branch2_conv(branch2_output)  # output tensor shape: (1, 32, 16, 16)
        branch2_output = self.branch2_transpose_conv(branch2_output)  # output tensor shape: (1, 16, 32, 32)

        # Branch 3
        branch3_output = self.branch3_pool(x)  # output tensor shape: (1, 16, 16, 16)
        branch3_output = self.branch3_conv(branch3_output)  # output tensor shape: (1, 32, 16, 16)
        branch3_output = self.branch3_transpose_conv(branch3_output)  # output tensor shape: (1, 16, 32, 32)

        # Concatenation
        combined_output = torch.cat((branch1_output, branch2_output, branch3_output), dim=1)  # output tensor shape: (1, 64, 32, 32)

        # Refinement
        combined_output = self.concat_conv(combined_output)

        # Flatten and Fully Connected Layer
        combined_output = combined_output.view(combined_output.size(0), -1)
        output = self.fc(combined_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)