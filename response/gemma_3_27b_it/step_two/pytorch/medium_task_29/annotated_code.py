import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=1, stride=1),
            nn.Flatten()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Flatten()
        )
        self.fc1 = nn.Linear(in_features='mask', out_features=128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x1 = self.layer1(x)  # output tensor shape: (1, 16384)
        x2 = self.layer2(x)  # output tensor shape: (1, 8192)
        x3 = self.layer3(x)  # output tensor shape: (1, 4096)
        x = torch.cat((x1, x2, x3), dim=1)  # output tensor shape: (1, 28672)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)