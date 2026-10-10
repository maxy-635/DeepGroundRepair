import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1x1_1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.dropout1 = nn.Dropout(p=0.2)
        self.conv1x1_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.dropout2 = nn.Dropout(p=0.2)
        self.conv3x1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.dropout3 = nn.Dropout(p=0.2)
        self.conv1x3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=3)
        self.dropout4 = nn.Dropout(p=0.2)
        self.conv1x1_restore = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=1 * 28 * 28, out_features=10) # Assuming MNIST input size 28x28

    def forward(self, x):
        x = self.conv1x1_1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.conv1x1_2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.conv3x1(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.conv1x3(x)
        x = F.relu(x)
        x = self.dropout4(x)
        x = self.conv1x1_restore(x)
        
        x = x + torch.nn.functional.interpolate(x, size=(28*28), mode='linear', align_corners=False)
        x = self.flatten(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28)  # Batch size 1, 1 channel, sequence length 28 (for MNIST)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)