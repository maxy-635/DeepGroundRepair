import torch
import torch.nn as nn
import torch.optim as optim

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # the initial shape of x is: (4, 3, 32, 32)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Define the input shape for CIFAR-10 (3 channels, 32x32x3)
    input_size = 32
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, input_size, input_size)
    model = DL_Model()
    output_tensor = model(dummy_input)
    print("Output shape:", output_tensor.shape)