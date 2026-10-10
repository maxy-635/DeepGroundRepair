import torch
import torch.nn as nn
import torch.utils.data as Dataset
import torch.optim as optim

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.25)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features='mask', out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        x = self.conv1(x)  # output tensor shape: (1, 32, 28, 28)
        x = self.relu1(x)  # output tensor shape: (1, 32, 28, 28)
        x = self.conv2(x)  # output tensor shape: (1, 64, 28, 28)
        x = self.relu2(x)  # output tensor shape: (1, 64, 28, 28)
        x = self.avgpool1(x)  # output tensor shape: (1, 64, 14, 14)
        x = self.dropout(x)  # output tensor shape: (1, 64, 14, 14)
        x = self.conv3(x)  # output tensor shape: (1, 128, 14, 14)
        x = self.relu3(x)  # output tensor shape: (1, 128, 14, 14)
        x = self.conv4(x)  # output tensor shape: (1, 256, 14, 14)
        x = self.relu4(x)  # output tensor shape: (1, 256, 14, 14)
        x = self.avgpool2(x)  # output tensor shape: (1, 256, 7, 7)
        x = self.flatten(x)  # output tensor shape: (1, 12544)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)