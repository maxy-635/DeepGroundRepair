import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 1), stride=(2, 2))
        self.depthwise_conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(2, 2), groups=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), stride=(2, 2))
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc = nn.Linear(in_features=130, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.depthwise_conv(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # MNIST image shape: (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(model)