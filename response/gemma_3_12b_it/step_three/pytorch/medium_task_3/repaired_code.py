import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Convolutional layer followed by max pooling
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Block 2: Convolutional layer followed by max pooling
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layer for classification
        self.fc = nn.Linear(in_features=64 * 7 * 7, out_features=10)

    def forward(self, x):
        # Pass through Block 1
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)

        # Pass through Block 2
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        # Flatten the output for the fully connected layer
        x = x.view(-1, 64 * 7 * 7)

        # Classification using the fully connected layer
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (1, 28, 28) - MNIST image size
    input_tensor = torch.randn(1, 1, 28, 28)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)