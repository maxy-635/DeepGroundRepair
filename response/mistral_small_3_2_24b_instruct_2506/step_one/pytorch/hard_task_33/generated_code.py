import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.branch1 = self._make_branch(64)
        self.branch2 = self._make_branch(64)
        self.branch3 = self._make_branch(64)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 3 * 7 * 7, 10)  # MNIST has 10 classes

    def _make_branch(self, channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # MNIST image shape: (1, 1, 28, 28)
    output_tensor = model(input_tensor)
    print(output_tensor)