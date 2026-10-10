import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=35, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=101, out_channels=128, kernel_size=3, stride=1, padding=1)
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
        # First fully connected layer
        self.fc1 = nn.Linear(in_features=128*32*32, out_features=512)
        # Second fully connected layer
        self.fc2 = nn.Linear(in_features=512, out_features=10)
        # Softmax activation
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # First convolutional layer with ReLU
        x1 = self.relu(self.conv1(x))
        # Concatenate input and output of first layer
        x = torch.cat((x, x1), dim=1)
        # Second convolutional layer with ReLU
        x2 = self.relu(self.conv2(x))
        # Concatenate input and output of second layer
        x = torch.cat((x, x2), dim=1)
        # Third convolutional layer with ReLU
        x3 = self.relu(self.conv3(x))
        # Concatenate input and output of third layer
        x = torch.cat((x, x3), dim=1)
        # Flatten the output
        x = torch.flatten(x, start_dim=1)
        # First fully connected layer with ReLU
        x = self.relu(self.fc1(x))
        # Second fully connected layer with Softmax
        x = self.softmax(self.fc2(x))
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)