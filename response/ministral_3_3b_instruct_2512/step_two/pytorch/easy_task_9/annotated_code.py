import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1)
        self.depthwise_conv = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1)
        self.add = nn.Add()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=16*28*28, out_features=10)

    def forward(self):
        x = self.conv1(x)
        x = self.depth1(x)
        x = self.conv2(x)
        x = self.add(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # MNIST input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)