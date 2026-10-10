import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: Split input into three groups along the last dimension
        # Group 1: 1x1 convolution
        self.group1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Group 2: 3x3 convolution
        self.group2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Group 3: 5x5 convolution
        self.group3 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Concatenate the outputs of the three groups
        self.concat = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Second block: Multiple branches for feature extraction
        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Branch 2: <1x1 convolution, 3x3 convolution>
        self.branch2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Branch 3: <1x1 convolution, 3x3 convolution>
        self.branch3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Branch 4: <max pooling, 1x1 convolution>
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=1, stride=1, padding=0),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Concatenate the outputs of all branches
        self.fuse = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 32 * 32, 10)  # Assuming input size is 32x32

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First block
        x1 = self.group1(x)  # output tensor shape: (1, 32, 32, 32)
        x2 = self.group2(x)  # output tensor shape: (1, 32, 32, 32)
        x3 = self.group3(x)  # output tensor shape: (1, 32, 32, 32)
        x = torch.cat((x1, x2, x3), dim=1)  # output tensor shape: (1, 96, 32, 32)
        x = self.concat(x)  # output tensor shape: (1, 64, 32, 32)

        # Second block
        x1 = self.branch1(x)  # output tensor shape: (1, 32, 32, 32)
        x2 = self.branch2(x)  # output tensor shape: (1, 32, 32, 32)
        x3 = self.branch3(x)  # output tensor shape: (1, 32, 32, 32)
        x4 = self.branch4(x)  # output tensor shape of this layer should be (1, 32, 32, 32)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.fuse(x)

        # Final layers
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Assuming CIFAR-10 images are 32x32
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)