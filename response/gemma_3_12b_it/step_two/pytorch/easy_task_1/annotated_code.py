import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.sequential1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.sequential2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.sequential1(x)
        x = self.sequential2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Dummy input tensor (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)