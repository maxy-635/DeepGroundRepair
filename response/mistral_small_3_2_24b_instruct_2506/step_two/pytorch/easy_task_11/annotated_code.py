import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=3, padding=0)
        self.conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features='mask', out_features=128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        x = self.avg_pool(x)  # output tensor shape: (1, 1, 8, 8)
        x = self.conv(x)  # output tensor shape: (1, 32, 8, 8)
        x = self.flatten(x)  # output tensor shape: (1, 2048)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # MNIST images are 28x28 grayscale
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])