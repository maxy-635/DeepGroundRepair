import torch
import torch.nn as nn
import torch.optim as optim

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(32 * 1 * 1, 10)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(10, 10)
        self.sigmoid = nn.functional.sigmoid

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.adaptive_pool(x)
        x = x.view(-1, 32 * 1 * 1)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)