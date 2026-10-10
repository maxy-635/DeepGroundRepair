import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Split input into three groups
        self.group1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.group2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.group3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )

        # Concatenate the outputs
        self.concat = nn.Sequential()

        # Flatten and classify
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=32 * 32 * 32, out_features=10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        # Split input
        x1 = self.group1(x)
        x2 = self.group2(x)
        x3 = self.group3(x)

        # Concatenate
        x = torch.concat([x1, x2, x3], dim=1)

        # Flatten and classify
        x = self.flatten(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 32, 32)  # CIFAR-10 images are 32x32
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])