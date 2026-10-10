import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Split into three groups with different kernel sizes
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # Second block: Multiple branches for feature extraction
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.branch5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=64 * 5 * 32 * 32, out_features=10)  # Assuming input size is 32x32

    def forward(self, x):
        # First block
        x1 = self.conv1x1(x)
        x2 = self.conv3x3(x)
        x3 = self.conv5x5(x)
        x = torch.cat([x1, x2, x3], dim=1)  # Concatenate along channel dimension

        # Second block
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        branch5 = self.branch5(x)
        x = torch.cat([branch1, branch2, branch3, branch4, branch5], dim=1)

        # Final layers
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])