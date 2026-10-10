import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.dropout3 = nn.Dropout(p=0.2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.dropout4 = nn.Dropout(p=0.2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = F.relu(self.conv3(x))
        x = self.dropout3(x)
        x = F.relu(self.conv4(x))
        x = self.dropout4(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Example input shape: batch_size, channels, height, width
    output_tensor = model(input_tensor)
    print(output_tensor.shape)