import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # 1x1 convolutional layer to increase dimensionality
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1, padding=0)
        
        # 3x3 depthwise separable convolutional layer
        self.depthwise_conv = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, groups=16)

        # 1x1 convolutional layer to reduce dimensionality
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, stride=1, padding=0)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layer
        self.linear = nn.Linear(in_features=28*28, out_features=10)

    def forward(self, x):
        # Pass through the layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.depthwise_conv(x))
        x = self.conv2(x)

        # Add the output to the original input
        x = x + x  # Corrected to add output to the original input x
        
        # Flatten and pass through the linear layer
        x = self.flatten(x)
        x = self.linear(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)