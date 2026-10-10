import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features='mask', out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x1 = self.conv1x1(x)  # output tensor shape: (1, 64, 32, 32)
        x1 = F.relu(x1)  # output tensor shape: (1, 64, 32, 32)
        x2 = self.conv3x3(x)  # output tensor shape: (1, 64, 32, 32)
        x2 = F.relu(x2)  # output tensor shape: (1, 64, 32, 32)
        x3 = self.conv5x5(x)  # output tensor shape: (1, 64, 32, 32)
        x3 = F.relu(x3)  # output tensor shape: (1, 64, 32, 32)
        x_concat = torch.cat((x1, x2, x3), dim=1)  # output tensor shape: (1, 192, 32, 32)
        x_flatten = self.flatten(x_concat)  # output tensor shape: (1, 196608)
        x_fc1 = self.fc1(x_flatten)
        x_fc1 = F.relu(x_fc1)
        x_out = self.fc2(x_fc1)
        return x_out


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)