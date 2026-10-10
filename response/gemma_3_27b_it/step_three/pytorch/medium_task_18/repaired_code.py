import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        conv1x1_out = self.relu(self.conv1x1(x))  # output tensor shape: (1, 16, 32, 32)
        conv3x3_out = self.relu(self.conv3x3(x))  # output tensor shape: (1, 32, 32, 32)
        conv5x5_out = self.relu(self.conv5x5(x))  # output tensor shape: (1, 64, 32, 32)

        combined_features = torch.concatenate([conv1x1_out, conv3x3_out, conv5x5_out], dim=1)  # output tensor shape: (1, 112, 32, 32)
        pooled_features = self.pool(combined_features)  # output tensor shape: (1, 112, 16, 16)

        flattened_features = self.flatten(pooled_features)  # output tensor shape: (1, 28672)
        fc1_out = self.relu(self.fc1(flattened_features))
        output = self.fc2(fc1_out)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)