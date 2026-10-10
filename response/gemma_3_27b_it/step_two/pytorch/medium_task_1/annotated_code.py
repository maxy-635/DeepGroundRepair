import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features='mask', out_features=128)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Convolutional layers and pooling
        x = self.conv1(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.relu1(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.conv2(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.relu2(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.pool(x)  # output tensor shape: (1, 32, 16, 16)

        # Add input to features
        x = x + x  # output tensor shape: (1, 32, 16, 16)

        # Flatten and fully connected layers
        x = self.flatten(x)  # output tensor shape: (1, 8192)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output: torch.Size([1, 10])