import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.pool1_1 = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.pool1_2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.pool1_4 = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        self.flatten1_1 = nn.Flatten()
        self.flatten1_2 = nn.Flatten()
        self.flatten1_4 = nn.Flatten()
        self.dropout1 = nn.Dropout(p=0.5)

        # Fully connected layer and reshaping
        self.fc1 = nn.Linear(in_features=1568, out_features=128)  # Adjusted input feature size
        self.reshape1 = nn.Reshape(target_shape='mask')

        # Block 2
        self.conv2_1 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2_2_1 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2_2_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_3_1 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2_3_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2_3_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2_1 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv2_4 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Final fully connected layers
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # Block 1
        x1_1 = self.pool1_1(x)
        x1_2 = self.pool1_2(x)
        x1_4 = self.pool1_4(x)

        x1_1 = self.flatten1_1(x1_1)
        x1_2 = self.flatten1_2(x1_2)
        x1_4 = self.flatten1_4(x1_4)

        x1_1 = self.dropout1(x1_1)
        x1_2 = self.dropout1(x1_2)
        x1_4 = self.dropout1(x1_4)

        x_block1 = torch.cat([x1_1, x1_2, x1_4], dim=1)

        # Fully connected layer and reshaping
        x = self.fc1(x_block1)
        x = F.relu(x)
        x = self.reshape1(x)

        # Block 2
        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2_1(x)
        x2_2 = self.conv2_2_2(x2_2)
        x2_3 = self.conv2_3_1(x)
        x2_3 = self.conv2_3_2(x2_3)
        x2_3 = self.conv2_3_3(x2_3)
        x2_4 = self.pool2_1(x)
        x2_4 = self.conv2_4(x2_4)

        x_block2 = torch.cat([x2_1, x2_2, x2_3, x2_4], dim=1)

        # Final fully connected layers
        x = F.relu(x_block2)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)