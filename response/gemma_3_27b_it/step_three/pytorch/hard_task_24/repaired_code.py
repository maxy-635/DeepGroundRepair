import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial 1x1 convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Branch 1: Local Feature Extraction
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Branch 2: Downsampling + Convolution + Upsampling
        self.branch2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        # Branch 3: Downsampling + Convolution + Upsampling
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        # Fusion and Final Convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels=32*3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # Initial Convolution
        x = self.initial_conv(x)

        # Branch 1
        branch1_output = self.branch1(x)

        # Branch 2
        branch2_output = self.branch2(x)

        # Branch 3
        branch3_output = self.branch3(x)

        # Concatenation
        x = torch.cat((branch1_output, branch2_output, branch3_output), dim=1)

        # Fusion Convolution
        x = self.fusion_conv(x)

        # Flatten for Fully Connected Layers
        x = x.view(x.size(0), -1)

        # Fully Connected Layers
        x = nn.ReLU(inplace=True)(self.fc1(x))
        x = nn.ReLU(inplace=True)(self.fc2(x))
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)