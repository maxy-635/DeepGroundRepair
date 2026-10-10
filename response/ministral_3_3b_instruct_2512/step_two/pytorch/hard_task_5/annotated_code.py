import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Split into three groups, each processed by a 1x1 convolutional layer
        self.block1_conv1 = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block1_conv2 = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.block1_conv3 = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Concatenate the outputs of the three convolutions
        self.block1_concat = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Block 2: Reshape and permute for channel shuffling
        self.block2_reshape = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Permute dimensions to achieve channel shuffling
        self.permute = nn.Sequential(
            nn.Permute((0, 2, 3, 1))
        )

        # Block 3: Depthwise separable convolution
        self.block3_conv = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Branch for direct connection to input
        self.branch = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Fully connected layer for classification
        self.fc = nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        # Block 1: Split and process
        x1 = self.block1_conv1(x)
        x2 = self.block1_conv2(x)
        x3 = self.block1_conv3(x)

        # Concatenate along channel dimension
        x_concat = torch.cat((x1, x2, x3), dim=1)
        x_concat = self.block1_concat(x_concat)

        # Block 2: Reshape and permute
        # Reshape to get shape (batch_size, channels, height, width)
        batch_size, channels, height, width = x_concat.shape
        x_reshape = x_concat.view(batch_size, channels, height, width)

        # Reshape into three groups
        x_grouped = x_reshape.view(batch_size, 3, (channels // 3), height, width)

        # Permute dimensions to achieve channel shuffling
        x_shuffled = self.permute(x_grouped)

        # Reshape back to original shape
        x_shuffled = x_shuffled.view(batch_size, channels, height, width)

        # Block 3: Depthwise separable convolution
        x_depthwise = self.block3_conv(x_shuffled)

        # Branch: Direct connection to input
        branch_output = self.branch(x)

        # Add outputs from main path and branch
        x_add = torch.add(x_depthwise, branch_output)

        # Fully connected layer for classification
        x_flat = x_add.view(batch_size, -1)
        output = self.fc(x_flat)

        return output

if __name__ == "__main__":
    # Dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 32, 32)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)