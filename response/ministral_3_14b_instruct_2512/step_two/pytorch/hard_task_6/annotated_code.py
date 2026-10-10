import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Split into 3 groups with 1x1 conv and ReLU
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask'),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Block 2: Channel shuffling
        self.block2 = nn.Sequential(
            nn.Unfold(kernel_size=1, padding=0, stride=1),
            nn.Fold(output_size=(32, 32), kernel_size=1, padding=0),
            nn.Unfold(kernel_size=1, padding=0, stride=1),
            nn.Fold(output_size=(32, 32), kernel_size=1, padding=0)
        )

        # Block 3: 3x3 depthwise separable convolution
        self.block3 = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Branch path: average pooling
        self.branch = nn.Sequential(
            nn.AvgPool2d(kernel_size=32, stride=1, padding=0)
        )

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3, 10)  # Assuming output is 10 classes for CIFAR-10

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path
        x_block1 = self.block1(x)
        x_block2 = self.block2(x_block1)
        x_block3 = self.block3(x_block2)

        # Branch path
        x_branch = self.branch(x)

        # Concatenate main path and branch path
        x_concat = torch.cat((x_block3, x_branch), dim=1)

        # Final classification
        x_flat = self.flatten(x_concat)
        x_out = self.fc(x_flat)

        return x_out

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)