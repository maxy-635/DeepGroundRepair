import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels='mask', kernel_size=3, stride=1, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size='mask', stride='mask', padding='mask')
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
        # Flatten layer
        self.flatten = nn.Flatten()
        # First fully connected layer
        self.fc1 = nn.Linear(in_features=64*8*8, out_features=128)
        # Second fully connected layer
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        # Softmax activation
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Input addition (skip connection)
        input_add = x  # output tensor shape: (1, 3, 32, 32)
        # First convolutional layer with ReLU
        x = self.relu(self.conv1(x))  # output tensor shape: (1, 32, 32, 32)
        # Second convolutional layer with ReLU
        x = self.relu(self.conv2(x))  # output tensor shape of this layer should be (1, 3, 32, 32)
        # Max pooling
        x = self.pool(x)  # output tensor shape of this layer should be (1, 64, 32, 32)
        # Add input to the output features
        x = x + input_add
        # Flatten the output
        x = self.flatten(x)
        # First fully connected layer with ReLU
        x = self.relu(self.fc1(x))
        # Second fully connected layer with Softmax
        x = self.softmax(self.fc2(x))
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])