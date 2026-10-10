import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Depthwise separable convolution: depthwise conv + 1x1 conv
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, groups=1),  # Depthwise conv
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0),  # 1x1 conv
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )

        # Additional convolutional layers (optional, can be added if needed)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.25),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )

        # Flatten and fully connected layer
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64*28*28, out_features=10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        x = self.depthwise_conv(x)  # output tensor shape: (1, 32, 28, 28)
        x = self.conv_layers(x)  # output tensor shape: (1, 64, 28, 28)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    # MNIST images are 28x28 with 1 channel
    input_tensor = torch.randn(1, 1, 28, 28)  # batch_size=1
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # Should be [1, 10]