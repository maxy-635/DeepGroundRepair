import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Split the input into three groups along the last dimension
        self.split = nn.ModuleList([
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        ])

        # Depthwise separable convolution layers
        self.depthwise_conv = nn.ModuleList([
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, groups=32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2, groups=32)
        ])

        # Pointwise convolution layers
        self.pointwise_conv = nn.ModuleList([
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)
        ])

        # Fully connected layer for classification
        self.fc = nn.Linear(in_features=64 * 8 * 8, out_features=10)  # Assuming input size is 32x32 and after convolutions it's 8x8

    def forward(self, x):
        # Split the input into three groups
        x1 = self.split[0](x)
        x2 = self.split[1](x)
        x3 = self.split[2](x)

        # Apply depthwise separable convolutions
        x1 = self.depthwise_conv[0](x1)
        x1 = self.pointwise_conv[0](x1)

        x2 = self.depthwise_conv[1](x2)
        x2 = self.pointwise_conv[1](x2)

        x3 = self.depthwise_conv[2](x3)
        x3 = self.pointwise_conv[2](x3)

        # Concatenate the outputs
        x = torch.cat([x1, x2, x3], dim=1)

        # Flatten and pass through the fully connected layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])