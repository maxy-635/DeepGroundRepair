import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First 1x1 convolutional layer to increase dimensionality
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        # 3x3 depthwise separable convolutional layer for feature extraction
        self.depthwise_conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32)
        # Second 1x1 convolutional layer to reduce dimensionality
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
        # Flattening layer
        self.flatten = nn.Flatten()
        # Fully connected layer for classification
        self.fc = nn.Linear(in_features=28*28, out_features=10)

    def forward(self, x):
        # Save the original input for residual connection
        residual = x
        # Apply the first 1x1 convolution
        x = self.conv1(x)
        # Apply the depthwise separable convolution
        x = self.depthwise_conv(x)
        # Apply the second 1x1 convolution
        x = self.conv2(x)
        # Add the residual connection
        x = x + residual
        # Flatten the output
        x = self.flatten(x)
        # Apply the fully connected layer
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor)