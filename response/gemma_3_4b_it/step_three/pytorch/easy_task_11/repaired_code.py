import torch
import torch.nn as nn
import torch.utils.data as data

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=4096, out_features=128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        x = self.avg_pool(x)  # output tensor shape: (1, 1, 8, 8)
        x = self.conv1(x)  # output tensor shape: (1, 64, 8, 8)
        x = self.relu1(x)  # output tensor shape: (1, 64, 8, 8)
        x = self.conv2(x)  # output tensor shape: (1, 64, 8, 8)
        x = self.relu2(x)  # output tensor shape: (1, 64, 8, 8)
        x = self.flatten(x)  # output tensor shape: (1, 4096)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)