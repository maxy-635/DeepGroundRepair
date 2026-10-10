import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=8, stride=8)
        self.fc1 = nn.Linear(32 * 8 * 8, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        branch1 = self.conv1(x)
        branch1 = self.relu1(branch1)
        branch2 = self.conv2(x)
        branch2 = self.relu2(branch2)
        combined = torch.add(branch1, branch2)
        pooled = self.avgpool(combined)
        flattened = pooled.view(-1, 32 * 8 * 8)
        output = self.fc1(flattened)
        output = self.softmax(output)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)