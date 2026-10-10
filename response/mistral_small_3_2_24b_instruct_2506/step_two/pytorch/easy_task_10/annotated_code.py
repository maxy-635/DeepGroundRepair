import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1), stride=(2, 2))
        self.depthwise_conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), groups=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), stride=(2, 2))
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc = nn.Linear(in_features='mask', out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        x = self.conv1(x)  # output tensor shape: (1, 32, 14, 14)
        x = self.depthwise_conv(x)  # output tensor shape: (1, 32, 6, 6)
        x = self.conv2(x)  # output tensor shape: (1, 10, 3, 3)
        x = self.flatten(x)  # output tensor shape: (1, 90)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # MNIST image shape: (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(model)