import torch
import torch.nn as nn
import torch.optim as optim

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Define fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Concatenation with previous layer's input
        x1 = self.conv1(x)  # output tensor shape: (1, 16, 32, 32)
        x1_concat = torch.concatenate([x, x1], dim=1)  # output tensor shape: (1, 19, 32, 32)

        x2 = self.conv2(x1_concat)
        x2_concat = torch.concatenate([x1_concat, x2], dim=1)

        x3 = self.conv3(x2_concat)
        x3_concat = torch.concatenate([x2_concat, x3], dim=1)

        # Flatten the output from the convolutional layers
        x_flatten = torch.flatten(x3_concat, start_dim=1)

        # Pass through fully connected layers
        x_fc1 = torch.relu(self.fc1(x_flatten))
        x_fc2 = self.fc2(x_fc1)

        # Apply softmax to get classification probabilities
        output = self.softmax(x_fc2)

        return output


if __name__ == "__main__":
    model = DL_Model()

    # Dummy input tensor (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 32, 32)

    output_tensor = model(input_tensor)

    print("Input tensor shape:", input_tensor.shape)
    print("Output tensor shape:", output_tensor.shape)