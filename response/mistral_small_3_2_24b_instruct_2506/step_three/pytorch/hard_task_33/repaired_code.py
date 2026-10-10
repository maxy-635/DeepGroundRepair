import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.branch1 = self._make_branch(64)
        self.branch2 = self._make_branch(64)
        self.branch3 = self._make_branch(64)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=150528, out_features=10)

    def _make_branch(self, channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        x1 = self.branch1(x)  # output tensor shape: (1, 64, 28, 28)
        x2 = self.branch2(x)  # output tensor shape: (1, 64, 28, 28)
        x3 = self.branch3(x)  # output tensor shape: (1, 64, 28, 28)
        x = torch.cat([x1, x2, x3], dim=1)  # output tensor shape: (1, 192, 28, 28)
        x = self.flatten(x)  # output tensor shape: (1, 150528)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # MNIST image shape: (1, 1, 28, 28)
    output_tensor = model(input_tensor)
    print(output_tensor)