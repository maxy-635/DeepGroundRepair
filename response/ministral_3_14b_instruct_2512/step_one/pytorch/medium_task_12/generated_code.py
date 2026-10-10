import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(num_features=32)
        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(num_features=32)

        # Block 2
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(num_features=64)
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(num_features=64)

        # Block 3
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(num_features=128)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatcchNorm2d(num_features=128)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=128*4*4, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # Block 1
        x1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x1 = F.relu(self.bn1_2(self.conv1_2(x1)))

        # Block 2
        x2 = torch.cat([x1, x1], dim=1)  # Concatenate along channel dimension
        x2 = F.relu(self.bn2_1(self.conv2_1(x2)))
        x2 = F.relu(self.bn2_2(self.conv2_2(x2)))

        # Block 3
        x3 = torch.cat([x2, x2], dim=1)  # Concatenate along channel dimension
        x3 = F.relu(self.bn3_1(self.conv3_1(x3)))
        x3 = F.relu(self.bn3_2(self.conv3_2(x3)))

        # Flatten and fully connected layers
        x = x3.view(x3.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)