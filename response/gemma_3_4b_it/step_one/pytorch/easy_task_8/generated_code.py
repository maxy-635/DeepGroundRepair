import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(p=0.25)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(p=0.25)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)