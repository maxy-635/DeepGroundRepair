import torch
import torch.nn as nn
import torch.utils.data

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=1024, out_features=128)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x = self.conv1(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.relu1(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.pool1(x)  # output tensor shape: (1, 32, 16, 16)

        x = self.conv2(x)  # output tensor shape: (1, 64, 16, 16)
        x = self.relu2(x)  # output tensor shape: (1, 64, 16, 16)
        x = self.pool2(x)  # output tensor shape: (1, 64, 8, 8)

        x = self.conv3(x)  # output tensor shape: (1, 64, 8, 8)
        x = self.relu3(x)  # output tensor shape: (1, 64, 8, 8)
        x = self.pool3(x)  # output tensor shape: (1, 64, 4, 4)

        x = self.flatten(x)  # output tensor shape: (1, 1024)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)