import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self, num_classes=10):
        super(DL_Model, self).__init__()

        # Channel compression with 1x1 convolution
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True)
        )

        # Parallel convolutional layers
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True)
        )

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True)
        )

        # Adaptive pooling to reduce spatial dimensions
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=32, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        # Channel compression
        x = self.compress(x)

        # Parallel convolutions and concatenation
        conv1x1_out = self.conv1x1(x)
        conv3x3_out = self.conv3x3(x)
        x = torch.cat([conv1x1_out, conv3x3_out], dim=1)

        # Adaptive pooling
        x = self.pool(x)

        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 64, 32, 32)  # Batch size 1, 64 channels, 32x32 spatial dimensions
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)