import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 10)
        self.reshape = nn.Reshape((10, 1, 1, 10))
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.gap(x)
        x = self.reshape(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)