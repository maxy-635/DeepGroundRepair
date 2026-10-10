import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)

        self.block1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        )

        self.conv2 = nn.Conv2d(32, 10, kernel_size=1, stride=1, padding=0)
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features='mask', out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        input_path = self.conv1(x)  # output tensor shape: (1, 16, 32, 32)
        input_path = self.bn1(input_path)  # output tensor shape: (1, 16, 32, 32)
        input_path = self.relu1(input_path)  # output tensor shape: (1, 16, 32, 32)

        branch_path = self.branch(x)  # output tensor shape: (1, 16, 32, 32)

        merged = input_path + branch_path  # output tensor shape: (1, 16, 32, 32)
        merged = self.block1(merged)  # output tensor shape: (1, 16, 32, 32)

        pooled = self.avgpool(merged)  # output tensor shape: (1, 16, 8, 8)
        flattened = self.flatten(pooled)  # output tensor shape: (1, 1024)
        output = self.linear(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)