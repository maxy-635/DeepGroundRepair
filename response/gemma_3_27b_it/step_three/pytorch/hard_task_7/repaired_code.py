import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial Convolutional Layer
        self.initial_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Block 1
        self.block1_group1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        )

        # Block 2
        self.block2 = nn.Sequential()
        
        # Fully Connected Layer
        self.fc = nn.Linear(in_features=32 * 28 * 28, out_features=10)

    def forward(self, x):
        # Initial Convolution
        x = self.initial_conv(x)

        # Block 1
        x1, x2 = torch.split(x, x.shape[3] // 2, dim=3)
        x1 = self.block1_group1(x1)
        x = torch.cat((x1, x2), dim=3)

        # Block 2
        height, width = x.shape[2], x.shape[3]
        groups = 4
        channels = x.shape[1]
        channels_per_group = channels // groups

        x = torch.reshape(x, (x.shape[0], height, width, groups, channels_per_group))
        x = torch.permute(x, (0, 2, 3, 4, 1))
        x = torch.reshape(x, (x.shape[0], height, width, channels))

        # Flatten and Fully Connect
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)