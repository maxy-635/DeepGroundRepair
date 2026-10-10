import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self, in_channels=1, out_channels=10):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_path1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, groups=32, bias=False),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)
        )

        self.block1_path2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32, bias=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Block 2
        self.block2 = nn.Sequential()

        self.fc = nn.Linear(in_features=128 * 7 * 7, out_features=out_channels)

    def forward(self, x):
        # Block 1
        x1 = self.block1_path1(x)
        x2 = self.block1_path2(x)
        x = torch.cat((x1, x2), dim=1)

        # Block 2
        batch_size, channels, height, width = x.shape
        x = x.reshape(batch_size, channels, height, width)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, width, channels // 4, 4)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, channels, height, width)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)