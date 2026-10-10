import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First path: Three sequential convolutional layers with addition
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        # Second path: Direct convolutional layer
        self.conv_direct = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu_direct = nn.ReLU(inplace=True)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=128)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # First path
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        x2 = self.conv2(x1)
        x2 = self.relu2(x2)
        x3 = self.conv3(x2)
        x3 = self.relu3(x3)

        # Second path
        x_direct = self.conv_direct(x)
        x_direct = self.relu_direct(x_direct)
        
        # Addition
        x_combined = x3 + x_direct

        # Fully connected layers
        x_flattened = self.flatten(x_combined)
        x_fc1 = self.fc1(x_flattened)
        x_fc1 = self.relu_fc1(x_fc1)
        x_output = self.fc2(x_fc1)

        return x_output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)