import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        self.layer_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128)
        )
        self.layer_2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64)
        )
        self.layer_3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x = self.conv1(x)  # output tensor shape: (1, 16, 32, 32)
        x = torch.cat([x, x], dim=1)  # output tensor shape: (1, 32, 32, 32)
        x = self.conv2(x)
        x = torch.cat([x, x], dim=1)
        x = self.conv3(x)
        x = torch.cat([x, x], dim=1)

        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (batch_size=1, channels=3, height=32, width=32)
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)