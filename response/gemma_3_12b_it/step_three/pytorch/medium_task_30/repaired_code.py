import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.avgpool1 = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.avgpool3 = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=48, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=10)  # 10 classes for CIFAR-10

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x = self.avgpool1(x)  # output tensor shape: (1, 3, 32, 32)
        x = self.avgpool2(x)  # output tensor shape: (1, 3, 16, 16)
        x = self.avgpool3(x)  # output tensor shape: (1, 3, 4, 4)
        x = self.flatten(x)  # output tensor shape: (1, 48)
        x = torch.cat([x], dim=1)  # output tensor shape: (1, 48)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 input shape: (batch_size, channels, height, width)
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input: batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)