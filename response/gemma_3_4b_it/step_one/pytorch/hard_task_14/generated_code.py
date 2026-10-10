import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))

        self.linear1 = nn.Linear(64 * 1 * 1, 64)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(64, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Main path
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = x.view(-1, 64 * 1 * 1)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        x = self.sigmoid(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)