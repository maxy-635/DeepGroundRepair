import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolution layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)

        # First parallel block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.relu2 = nn.ReLU(inplace=True)

        # Second parallel block
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU(inplace=True)

        # Third parallel block
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=32)
        self.relu4 = nn.ReLU(inplace=True)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32768, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Initial convolution
        x = self.conv1(x)  # output tensor shape: (1, 32, 32, 32)

        # First parallel block
        block1 = self.conv2(x)  # output tensor shape: (1, 32, 32, 32)
        block1 = self.bn2(block1)  # output tensor shape: (1, 32, 32, 32)
        block1 = self.relu2(block1)  # output tensor shape: (1, 32, 32, 32)

        # Second parallel block
        block2 = self.conv3(x)  # output tensor shape: (1, 32, 32, 32)
        block2 = self.bn3(block2)  # output tensor shape: (1, 32, 32, 32)
        block2 = self.relu3(block2)  # output tensor shape: (1, 32, 32, 32)

        # Third parallel block
        block3 = self.conv4(x)  # output tensor shape: (1, 32, 32, 32)
        block3 = self.bn4(block3)  # output tensor shape: (1, 32, 32, 32)
        block3 = self.relu4(block3)  # output tensor shape: (1, 32, 32, 32)

        # Add outputs of blocks with initial convolution
        x = x + block1 + block2 + block3  # output tensor shape: (1, 32, 32, 32)

        # Flatten and pass through fully connected layers
        x = self.flatten(x)  # output tensor shape: (1, 32768)
        x = self.fc1(x)
        x = self.fc2(x)

        # Softmax activation
        x = torch.softmax(x, dim=1)
        return x

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 32, 32)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input tensor shape:", input_tensor.shape)
    print("Output tensor shape:", output_tensor.shape)