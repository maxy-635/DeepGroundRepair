import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Linear(in_features='mask', out_features=1024),
            nn.Unflatten(dim=1, unflattened_size=(1024, 1, 32, 32))
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
            ),
            nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            ),
            nn.Flatten(),
            nn.Linear(in_features=262144, out_features=10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        x = self.block1(x)
        x = self.block2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # MNIST image shape: (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(model)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)