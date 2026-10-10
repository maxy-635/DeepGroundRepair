import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0)
        self.depthwise_conv = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
        self.relu1 = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=8)
        self.fc1 = nn.Linear(in_features=512, out_features=10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x = self.conv1(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.relu1(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.depthwise_conv(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.relu1(x)  # output tensor shape: (1, 32, 32, 32)
        x = self.avg_pool(x)  # output tensor shape: (1, 32, 4, 4)
        x = torch.flatten(x, 1)  # output tensor shape: (1, 512)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)