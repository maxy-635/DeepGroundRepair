import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1x1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32 * 26 * 26, out_features=128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1x1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # batch_size=1, channels=1, height=28, width=28
    output_tensor = model(input_tensor)
    print(output_tensor)