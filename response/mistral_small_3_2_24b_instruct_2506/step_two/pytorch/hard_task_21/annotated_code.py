import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1x1 = nn.Conv2d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')
        self.conv3x3 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)

        # Branch path
        self.branch_conv = nn.Conv2d(in_channels=3, out_channels=192, kernel_size=1, stride=1, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=192 * 8 * 8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Split input into three groups along the channel dimension
        x1 = x[:, :1, :, :]  # output tensor shape: (1, 1, 32, 32)
        x2 = x[:, 1:2, :, :]  # output tensor shape: (1, 1, 32, 32)
        x3 = x[:, 2:, :, :]  # output tensor shape: (1, 1, 32, 32)

        # Main path
        out1 = self.conv1x1(x1)
        out2 = self.conv3x3(x2)
        out3 = self.conv5x5(x3)

        # Concatenate along the channel dimension
        main_out = torch.cat([out1, out2, out3], dim=1)

        # Branch path
        branch_out = self.branch_conv(x)

        # Add main and branch outputs
        combined = main_out + branch_out

        # Flatten and pass through fully connected layers
        combined = combined.view(combined.size(0), -1)
        combined = F.relu(self.fc1(combined))
        combined = self.fc2(combined)

        return combined

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # CIFAR-10 input shape: (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (batch_size, 10)