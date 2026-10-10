import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First 1x1 convolution to reduce dimensionality
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        # 3x3 convolution for feature extraction
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Second 1x1 convolution to restore dimensionality
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        # Flatten the output
        self.flatten = nn.Flatten()
        # Fully connected layer for classification
        self.fc = nn.Linear(in_features=32 * 7 * 7, out_features=10)

    def forward(self, x):
        # Apply the sequence of layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    # MNIST images are typically 28x28 pixels with 1 channel
    input_tensor = torch.randn(1, 1, 28, 28)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)