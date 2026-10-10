import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Split into three groups, apply 1x1 conv, and concatenate
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32)
        )

        # Block 2: Channel shuffling
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32)
        )

        # Block 3: 3x3 depthwise separable convolution
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32)
        )

        # Branch path: Average pooling
        self.branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        )

        # Final fully connected layer
        self.fc = nn.Linear(in_features=32 * 32 * 32, out_features=10)

    def forward(self, x):
        # Main path
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block1(x)  # Repeat Block 1

        # Branch path
        branch_out = self.branch(x)

        # Concatenate main path and branch path
        x = torch.cat((x, branch_out), dim=1)

        # Flatten and pass through fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Dummy input tensor for CIFAR-10 (batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)