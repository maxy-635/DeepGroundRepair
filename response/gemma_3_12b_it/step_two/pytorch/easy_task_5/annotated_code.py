import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features='mask', out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        x = self.conv1(x)  # output tensor shape: (1, 16, 28, 28)
        x = self.conv2(x)  # output tensor shape: (1, 32, 28, 28)
        x = self.conv3(x)  # output tensor shape: (1, 16, 28, 28)
        x = self.flatten(x)  # output tensor shape: (1, 12544)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Example input: Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)