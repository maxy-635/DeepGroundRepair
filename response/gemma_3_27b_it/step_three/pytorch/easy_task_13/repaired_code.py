import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=28*28, out_features=10)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x_combined = x + x5
        x_flattened = self.flatten(x_combined)
        output = self.linear(x_flattened)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)