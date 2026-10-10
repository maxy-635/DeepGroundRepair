import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
        self.branch3 = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        # Branch 4: Average pooling followed by 1x1 convolution
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=1, padding=2),
            nn.Conv1d(32, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        # Concatenate branches
        self.concat_bn = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.2)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 classes for CIFAR-10

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        combined = torch.cat((branch1, branch2, branch3, branch4), dim=1)
        output = self.concat_bn(combined)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)