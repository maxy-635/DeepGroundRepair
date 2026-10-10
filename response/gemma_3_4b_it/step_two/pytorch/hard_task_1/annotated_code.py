import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.avgpool1 = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc1_1 = nn.Linear(16 * 8 * 8, 128)
        self.fc1_2 = nn.Linear(16 * 8 * 8, 128)

        self.maxpool1 = nn.MaxPool2d(kernel_size=8, stride=1)
        self.fc2_1 = nn.Linear(16 * 8 * 8, 128)
        self.fc2_2 = nn.Linear(16 * 8 * 8, 128)

        # Channel attention
        self.add = nn.Add()
        self.sigmoid = nn.Sigmoid()

        # Block 2
        self.avgpool2 = nn.AvgPool2d(kernel_size=4, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=4, stride=1)
        self.concat = nn.Concatenate(dim=1)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=1, stride=1, padding=0)
        self.sigmoid2 = nn.Sigmoid()

        # Final classification
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.avgpool1(x)
        x = x.view(x.size(0), -1)
        x1 = self.fc1_1(x)
        x1 = nn.ReLU()(x1)
        x2 = self.fc1_2(x)
        x2 = nn.ReLU()(x2)

        x = self.maxpool1(x)
        x = x.view(x.size(0), -1)
        x3 = self.fc2_1(x)
        x3 = nn.ReLU()(x3)
        x4 = self.fc2_2(x)
        x4 = nn.ReLU()(x4)

        x = self.add(x1, x2)
        x = self.sigmoid(x)
        x = x * x

        # Block 2
        x = self.avgpool2(x)
        x = self.maxpool2(x)
        x = self.concat(x)
        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.sigmoid2(x)
        x = x * x

        # Add block 1 features
        x = self.add(x, x)

        # Final classification
        x = x.view(x.size(0), -1)
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)