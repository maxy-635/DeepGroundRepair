import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First part: 3 pairs of convolutional layer + max-pooling layer
        self.encoder = nn.Sequential(
            # First block
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second block
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third block
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Second part: convolutional layer + Dropout + convolutional layer
        self.middle = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Third part: 3 pairs of convolutional layer + transposed convolutional layer with skip connections
        self.decoder = nn.Sequential(
            # First upsampling block with skip connection from first encoder block
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),

            # Second upsampling block with skip connection from second encoder block
            nn.Conv2d(in_channels=256+128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),

            # Third upsampling block with skip connection from third encoder block
            nn.Conv2d(in_channels=128+64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )

        # Final 1x1 convolutional layer for classification
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # First part: feature extraction
        x1 = self.encoder[0:3](x)  # First block output
        x2 = self.encoder[3:6](x1)  # Second block output
        x3 = self.encoder[6:9](x2)  # Third block output

        # Second part: middle processing
        x = self.middle(x3)

        # Third part: upsampling with skip connections
        x = self.decoder[0:3](x)
        x = torch.cat([x, x3], dim=1)  # Skip connection
        x = self.decoder[3:6](x)
        x = torch.cat([x, x2], dim=1)  # Skip connection
        x = self.decoder[6:9](x)
        x = torch.cat([x, x1], dim=1)  # Skip connection

        # Final classification
        x = self.classifier(x)

        # Global average pooling to get class probabilities
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)