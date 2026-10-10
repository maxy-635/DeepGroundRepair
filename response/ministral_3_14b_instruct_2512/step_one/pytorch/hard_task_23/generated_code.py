import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial 1x1 convolution
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Local feature extraction branch
        self.local_branch = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Downsampling and upsampling branch 1
        self.downsample1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Downsampling and upsampling branch 2
        self.downsample2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Concatenation and refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Fully connected layer for classification
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 32 * 32, out_features=10)
        )

    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)

        # Local feature extraction branch
        local_features = self.local_branch(x)

        # Downsampling and upsampling branch 1
        downsample1_features = self.downsample1(x)

        # Downsampling and upsampling branch 2
        downsample2_features = self.downsample2(x)

        # Concatenate all features
        concatenated = torch.cat([local_features, downsample1_features, downsample2_features], dim=1)

        # Refinement
        x = self.refinement(concatenated)

        # Fully connected layer
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)