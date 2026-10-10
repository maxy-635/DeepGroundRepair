import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First 1x1 convolutional layer
        self.conv1x1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Dropout layer for first 1x1 convolution
        self.dropout1 = nn.Dropout2d(p=0.5)

        # Second 1x1 convolutional layer
        self.conv1x1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Dropout layer for second 1x1 convolution
        self.dropout2 = nn.Dropout2d(p=0.5)

        # 3x1 convolutional layer
        self.conv3x1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=1, padding=(1, 0))

        # Dropout layer for 3x1 convolution
        self.dropout3 = nn.Dropout2d(p=0.5)

        # 1x3 convolutional layer
        self.conv1x3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=1, padding=(0, 1))

        # Dropout layer for 1x3 convolution
        self.dropout4 = nn.Dropout2d(p=0.5)

        # 1x1 convolutional layer to restore original channel count
        self.conv1x1_restore = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layer
        self.fc = nn.Linear(in_features=784, out_features=10)

    def forward(self, x):
        # First 1x1 convolution
        x1 = self.conv1x1_1(x)
        x1 = self.dropout1(x1)

        # Second 1x1 convolution
        x2 = self.conv1x1_2(x1)
        x2 = self.dropout2(x2)

        # 3x1 convolution
        x3 = self.conv3x1(x2)
        x3 = self.dropout3(x3)

        # 1x3 convolution
        x4 = self.conv1x3(x3)
        x4 = self.dropout4(x4)

        # Restore original channel count
        x5 = self.conv1x1_restore(x4)

        # Add processed features to original input
        x = x + x5

        # Flatten and fully connected layer
        x = self.flatten(x)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)