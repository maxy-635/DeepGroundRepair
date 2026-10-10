import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(5, 5))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32 * 5 * 5, out_features=128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Example input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)