import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(5, 5))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=128, out_features=128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        x = self.avg_pool(x)  # output tensor shape: (1, 1, 5, 5)
        x = self.conv1(x)  # output tensor shape: (1, 32, 2, 2)
        x = self.flatten(x)  # output tensor shape: (1, 128)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Example input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)