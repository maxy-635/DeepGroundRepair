import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Define the max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Define the ReLU activation
        self.relu = nn.ReLU(inplace=True)

        # Define the fully connected layers
        self.fc1 = nn.Linear(in_features=64*8*8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

        # Define the softmax layer
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # First convolutional layer with ReLU
        x = self.conv1(x)
        x = self.relu(x)

        # Second convolutional layer with ReLU
        x = self.conv2(x)
        x = self.relu(x)

        # Max pooling layer
        x = self.pool(x)

        # Add the input layer to the output features
        x = x + self.conv1(x)

        # Flatten the output
        x = torch.flatten(x, 1)

        # First fully connected layer with ReLU
        x = self.fc1(x)
        x = self.relu(x)

        # Second fully connected layer
        x = self.fc2(x)

        # Apply softmax to get probability distribution
        x = self.softmax(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)