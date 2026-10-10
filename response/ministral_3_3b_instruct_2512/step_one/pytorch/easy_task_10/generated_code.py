import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # 1x1 convolutional layer to increase dimensionality
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=2)
        # 3x3 depthwise separable convolutional layer for feature extraction
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        # 1x1 convolutional layer to reduce dimensionality
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=2)
        # Flatten layer
        self.flatten = nn.Flatten()
        # Fully connected layer
        self.fc = nn.Linear(in_features=16*7*7, out_features=10)

    def forward(self, x):
        # Apply all layers in sequence
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # Create model instance
    model = DL_Model()

    # Create dummy input tensor with shape (batch_size, 1, 28, 28)
    input_tensor = torch.randn(1, 1, 28, 28)

    # Forward pass
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)