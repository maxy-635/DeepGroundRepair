import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Stage 1: Two convolution and max pooling layers for downsampling
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size='mask', stride='mask', padding='mask')
        )

        # Additional convolution and dropout layers
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        # Stage 2: Upsampling with skip connections
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.skip_conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.skip_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Final layers for classification
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 10, kernel_size=1, stride=1)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Stage 1: Downsampling
        x1 = self.conv1(x)  # output tensor shape: (1, 32, 16, 16)
        x2 = self.conv2(x1)  # output tensor shape of this layer should be (1, 64, 16, 16)

        # Additional convolution and dropout
        x3 = self.conv3(x2)  # output tensor shape: (1, 128, 8, 8)

        # Stage 2: Upsampling with skip connections
        up1 = self.upconv1(x3)  # output tensor shape: (1, 64, 16, 16)
        skip1 = self.skip_conv1(x2)
        up1 = up1 + skip1

        up2 = self.upconv2(up1)
        skip2 = self.skip_conv2(x1)
        up2 = up2 + skip2

        # Final classification layer
        output = self.final_conv(up2)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor with batch size 1, 3 channels, 32x32 spatial dimensions
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)