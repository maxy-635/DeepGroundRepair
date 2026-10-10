import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Depthwise separable convolution: depthwise conv followed by 1x1 conv
        self.depthwise_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, groups=1)
        self.pointwise_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.dropout1 = nn.Dropout(p=0.5)

        # Additional convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(p=0.5)

        # Flatten and fully connected layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=64*28*28, out_features=10)

    def forward(self, x):
        # Depthwise separable convolution
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        # Additional convolutional layer
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        # Flatten and fully connected layer
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # MNIST image shape: (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor)