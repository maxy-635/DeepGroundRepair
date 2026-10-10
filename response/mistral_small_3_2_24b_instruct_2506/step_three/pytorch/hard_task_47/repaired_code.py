import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Three groups with different kernel sizes
        self.group1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64)
        )

        self.group2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64)
        )

        self.group3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=64)
        )

        # Second block: Multiple branches for feature extraction
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=196608, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First block: Split and process
        x1 = self.group1(x)  # output tensor shape: (1, 64, 32, 32)
        x2 = self.group2(x)  # output tensor shape: (1, 64, 32, 32)
        x3 = self.group3(x)  # output tensor shape: (1, 64, 32, 32)
        x = torch.cat([x1, x2, x3], dim=1)  # output tensor shape: (1, 192, 32, 32)

        # Second block: Multiple branches
        y1 = self.branch1(x)  # output tensor shape: (1, 64, 32, 32)
        y2 = self.branch2(x)  # output tensor shape: (1, 64, 32, 32)
        y3 = self.branch3(x)  # output tensor shape: (1, 64, 32, 32)
        y = torch.cat([y1, y2, y3], dim=1)  # output tensor shape: (1, 192, 32, 32)

        # Flatten and fully connected layers
        y = y.view(y.size(0), -1)  # output tensor shape: (1, 196608)
        y = self.fc1(y)
        y = self.fc2(y)

        return y

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # CIFAR-10 input shape: (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (1, 10)