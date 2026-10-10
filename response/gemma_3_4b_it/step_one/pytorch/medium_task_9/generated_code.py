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
        self.linear = nn.Linear(10 * 4 * 4, 10)  # Assuming 10 classes

    def forward(self, x):
        input_path = self.conv1(x)
        input_path = self.bn1(input_path)
        input_path = self.relu1(input_path)

        branch_path = self.branch(x)

        merged = input_path + branch_path
        merged = self.block1(merged)

        pooled = self.avgpool(merged)
        flattened = self.flatten(pooled)
        output = self.linear(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)