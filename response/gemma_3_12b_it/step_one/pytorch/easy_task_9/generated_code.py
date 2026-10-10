import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # 1x1 convolutional layer to increase dimensionality
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        # 3x3 depthwise separable convolutional layer for feature extraction
        self.depthwise = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32)
        self.pointwise = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0)
        # Another 1x1 convolutional layer to reduce dimensionality
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=1, stride=1, padding=0)
        # Flatten layer
        self.flatten = nn.Flatten()
        # Fully connected layer
        self.linear = nn.Linear(in_features=8 * 7 * 7, out_features=10)  # Assuming input size of 28x28, after convolutions

    def forward(self, x):
        # 1x1 convolution
        x1 = self.conv1(x)
        # Depthwise separable convolution
        dw = self.depthwise(x1)
        pw = self.pointwise(dw)
        # 1x1 convolution to reduce dimensionality
        x2 = self.conv2(pw)
        # Add the processed output to the original input
        x = x + x2
        # Flatten the output
        x = self.flatten(x)
        # Fully connected layer
        x = self.linear(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (batch size 4, 1 channel, 28x28 image)
    input_tensor = torch.randn(4, 1, 28, 28)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)