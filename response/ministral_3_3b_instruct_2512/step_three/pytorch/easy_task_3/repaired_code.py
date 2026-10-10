import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First feature extraction block: two convolution layers followed by max pooling
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Second feature extraction block: three convolution layers followed by max pooling
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=512),
            nn.Linear(in_features=512, out_features=256),
            nn.Linear(in_features=256, out_features=10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        x = self.conv1(x)  # output tensor shape: (1, 64, 14, 14)
        x = self.conv2(x)  # output tensor shape: (1, 256, 7, 7)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 1, 28, 28)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)