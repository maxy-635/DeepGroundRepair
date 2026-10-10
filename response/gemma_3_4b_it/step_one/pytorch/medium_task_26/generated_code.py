import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv1d(64, 32, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.concat = nn.Concat(dim=1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(32 * 32, 16)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu1(x)
        x = self.conv3(x)
        x = self.concat(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(32, 32, 64)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)