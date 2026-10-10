import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features='mask', out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 64, 32, 32)
        x = self.conv1x1(x)  # output tensor shape: (1, 64, 32, 32)
        x = self.relu(x)  # output tensor shape: (1, 64, 32, 32)

        x1 = self.conv1x1_2(x)  # output tensor shape: (1, 64, 32, 32)
        x1 = self.relu(x1)  # output tensor shape: (1, 64, 32, 32)

        x2 = self.conv3x3(x)  # output tensor shape: (1, 64, 32, 32)
        x2 = self.relu(x2)  # output tensor shape: (1, 64, 32, 32)

        x = torch.cat((x1, x2), dim=1)  # output tensor shape: (1, 128, 32, 32)
        x = self.flatten(x)  # output tensor shape: (1, 131072)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 64, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)