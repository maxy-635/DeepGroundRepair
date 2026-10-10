import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Average pooling layers with different scales
        self.avg_pool1 = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.avg_pool4 = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)

        # Flatten layers
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()

        # Fully connected layer and reshape
        self.fc1 = nn.Linear(in_features=3*28*28, out_features=1024)
        self.reshape = nn.Sequential(
            nn.Flatten(),
            nn.Unflatten(1, (16, 8, 8))
        )

        # Second block: Depthwise separable convolutional layers
        self.conv1x1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=3)

        # Final fully connected layer
        self.fc2 = nn.Linear(in_features=64*8*8, out_features=10)

    def forward(self, x):
        # First block
        x1 = self.avg_pool1(x)
        x2 = self.avg_pool2(x)
        x3 = self.avg_pool4(x)

        x1 = self.flatten1(x1)
        x2 = self.flatten2(x2)
        x3 = self.flatten3(x3)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.fc1(x)
        x = self.reshape(x)

        # Second block
        x = torch.split(x, 4, dim=1)
        x1 = self.conv1x1(x[0])
        x2 = self.conv3x3(x[1])
        x3 = self.conv5x5(x[2])
        x4 = self.conv7x7(x[3])

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = torch.flatten(x, 1)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # MNIST image shape: (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor)