import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=128)
        self.dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)