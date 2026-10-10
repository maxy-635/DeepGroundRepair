import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 3x3 convolutions
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        # Branch 2: 1x1 conv followed by two 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 16, 1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        # Branch 3: Max pooling
        self.branch3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Multi-scale feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask'),
            nn.ReLU(inplace=True)
        )

        # Flatten
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x1 = self.branch1(x)  # output tensor shape: (1, 16, 16, 16)
        x2 = self.branch2(x)  # output tensor shape: (1, 32, 16, 16)
        x3 = self.branch3(x)  # output tensor shape: (1, 3, 16, 16)

        # Concatenate the branches
        x = torch.cat((x1, x2, x3), dim=1)  # output tensor shape: (1, 51, 16, 16)

        # Fusion
        x = self.fusion(x)

        # Flatten
        x = self.flatten(x)

        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)