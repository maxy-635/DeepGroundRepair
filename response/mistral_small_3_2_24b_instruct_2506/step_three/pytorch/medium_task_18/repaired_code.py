import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First convolutional block
        self.conv1x1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # Second convolutional block
        self.conv1x1_2 = nn.Conv2d(in_channels=195, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3_2 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5_2 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # Flatten and fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64*8*8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First convolutional block
        x1 = self.conv1x1(x)  # output tensor shape: (1, 64, 32, 32)
        x2 = self.conv3x3(x)  # output tensor shape: (1, 64, 32, 32)
        x3 = self.conv5x5(x)  # output tensor shape: (1, 64, 32, 32)
        x4 = self.maxpool(x)  # output tensor shape: (1, 3, 32, 32)
        x = torch.cat((x1, x2, x3, x4), dim=1)  # output tensor shape: (1, 195, 32, 32)

        # Second convolutional block
        x1 = self.conv1x1_2(x)
        x2 = self.conv3x3_2(x)
        x3 = self.conv5x5_2(x)
        x4 = self.maxpool_2(x)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        # Flatten and fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])