import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Encoder (Downsampling)
        self.encoder = nn.Sequential(
            # First convolutional block
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second convolutional block
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Additional convolutional and dropout layers
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.25)
        )

        # Decoder (Upsampling with skip connections)
        self.decoder = nn.Sequential(
            # First upsampling block with skip connection
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),

            # Second upsampling block with skip connection
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        # Final 1x1 convolution for classification
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoder path
        x = self.encoder(x)

        # Decoder path with skip connections
        # Here we use functional interpolation to upsample and concatenate with skip connections
        # For simplicity, we'll just pass through the decoder without explicit skip connections
        # In a real implementation, you'd need to store intermediate encoder outputs
        x = self.decoder(x)

        # Final classification layer
        x = self.final_conv(x)

        # Global average pooling to get a single probability vector
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)