import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First 1x1 convolutional layer
        self.conv1_1x1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)

        # First dropout layer
        self.dropout1 = nn.Dropout(p=0.2)

        # Second 1x1 convolutional layer
        self.conv2_1x1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Second dropout layer
        self.dropout2 = nn.Dropout(p=0.2)

        # 3x1 convolutional layer
        self.conv3_3x1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=1, padding=(1, 0))

        # Third dropout layer
        self.dropout3 = nn.Dropout(p=0.2)

        # 1x3 convolutional layer
        self.conv4_1x3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=1, padding=(0, 1))

        # Fourth dropout layer
        self.dropout4 = nn.Dropout(p=0.2)

        # Restore channels to match input using 1x1 convolutional layer
        self.conv5_1x1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layer
        self.fc = nn.Linear(in_features=64 * 28 * 28, out_features=10)

    def forward(self, x):
        # First 1x1 convolution
        x = self.conv1_1x1(x)

        # Add dropout
        x = self.dropout1(x)

        # Second 1x1 convolution
        x = self.conv2_1x1(x)

        # Add dropout
        x = self.dropout2(x)

        # 3x1 convolution
        x = self.conv3_3x1(x)

        # Add dropout
        x = self.dropout3(x)

        # 1x3 convolution
        x = self.conv4_1x3(x)

        # Add dropout
        x = self.dropout4(x)

        # Restore channels to match input
        x = self.conv5_1x1(x)

        # Add original input to processed features
        x = torch.add(x, self.conv1_1x1(x))

        # Flatten the tensor
        x = self.flatten(x)

        # Fully connected layer
        x = self.fc(x)

        return x

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 1, 28, 28)

    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)