import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Define the max pooling layers with different window sizes
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.pool3 = nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4))

        # Define ReLU activation
        self.relu = nn.ReLU(inplace=True)

        # Define flatten layers
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()

        # Define fully connected layers
        self.fc1 = nn.Linear(in_features=128*4*4, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First convolutional block
        x1 = self.relu(self.conv1(x))  # output tensor shape: (1, 32, 32, 32)
        x1 = self.pool1(x1)  # output tensor shape: (1, 32, 32, 32)
        x1 = self.flatten1(x1)  # output tensor shape: (1, 32768)

        # Second convolutional block
        x2 = self.relu(self.conv2(x1.view(-1, 32, 32, 32)))  # output tensor shape: (1, 64, 32, 32)
        x2 = self.pool2(x2)  # output tensor shape: (1, 64, 16, 16)
        x2 = self.flatten2(x2)  # output tensor shape: (1, 16384)

        # Third convolutional block
        x3 = self.relu(self.conv3(x2.view(-1, 64, 16, 16)))  # output tensor shape: (1, 128, 16, 16)
        x3 = self.pool3(x3)  # output tensor shape: (1, 128, 4, 4)
        x3 = self.flatten3(x3)  # output tensor shape: (1, 2048)

        # Concatenate the flattened features
        x_concat = torch.cat((x1, x2, x3), dim=1)  # output tensor shape: (1, 51200)

        # Fully connected layers
        x = self.relu(self.fc1(x_concat))
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)