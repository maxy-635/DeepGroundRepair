import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)

        # Flatten and concatenate
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Convolutional layers with max pooling
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)

        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        
        # Flatten and concatenate
        x = self.flatten1(x)
        x = self.flatten2(x)
        x = self.flatten3(x)
        
        # Concatenate the flattened tensors
        x = torch.cat((x, x, x), dim=1)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for CIFAR-10 (32x32 image, 3 channels)
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size = 1
    output_tensor = model(input_tensor)
    print(output_tensor.shape)