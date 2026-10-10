import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(32 * 32 * 32, 128)
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        # Feature extraction path 1
        x = self.conv1(x)
        x = self.relu1(x)

        # Feature extraction path 2
        x = self.conv2(x)
        x = self.relu2(x)

        # Feature extraction path 3
        x = self.conv3(x)

        # Concatenate the paths
        x = torch.cat([x, x], dim=1)

        # Align output dimensions
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)