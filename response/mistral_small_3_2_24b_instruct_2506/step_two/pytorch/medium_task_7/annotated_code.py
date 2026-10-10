import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu4 = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features='mask', out_features=1024, bias=True)
        self.fc2 = nn.Linear(in_features=1024, out_features=10, bias=True)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First path: conv1 -> relu1 -> conv2 -> relu2 -> conv3 -> relu3
        x1 = self.relu1(self.conv1(x))  # output tensor shape: (1, 64, 32, 32)
        x1 = self.relu2(self.conv2(x1))  # output tensor shape: (1, 64, 32, 32)
        x1 = self.relu3(self.conv3(x1))  # output tensor shape: (1, 64, 32, 32)

        # Second path: conv4 -> relu4
        x2 = self.relu4(self.conv4(x))  # output tensor shape: (1, 64, 32, 32)

        # Add outputs from both paths
        x = torch.add(x1, x2)  # output tensor shape: (1, 64, 32, 32)

        # Flatten and pass through fully connected layers
        x = self.flatten(x)  # output tensor shape: (1, 65536)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(model)