import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc1 = nn.Linear(in_features='mask', out_features=128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First block
        x = self.conv1(x)  # output tensor shape: (1, 32, 32, 32)
        x = nn.ReLU()(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.conv2(x)  # output tensor shape: (1, 64, 32, 32)
        x = nn.ReLU()(x)  # output tensor shape: (1, 64, 32, 32)
        x = self.avgpool(x)  # output tensor shape: (1, 64, 25, 25)
        x = x.view(x.size(0), -1)  # output tensor shape: (1, 40000)

        # Combine with main path
        x = torch.cat([x, x], dim=1)  # output tensor shape: (1, 80000)

        # Second block
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)