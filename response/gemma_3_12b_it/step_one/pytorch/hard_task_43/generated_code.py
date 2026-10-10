import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.avgpool1 = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.avgpool3 = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=16 * 16 * 3, out_features=128)  # Assuming input is 28x28x1
        self.reshape = nn.Reshape(shape=(128, 1, 1, 1))

        # Block 2
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

        self.concat = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1)
        )
        
        # Classification
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # Block 1
        x1 = self.avgpool1(x)
        x2 = self.avgpool2(x)
        x3 = self.avgpool3(x)

        x1 = self.flatten(x1.view(-1, 1, 28, 28))
        x2 = self.flatten(x2.view(-1, 1, 14, 14))
        x3 = self.flatten(x3.view(-1, 1, 7, 7))

        combined = torch.cat((x1, x2, x3), dim=1)
        combined = torch.relu(self.fc1(combined))
        combined = self.reshape(combined)

        # Block 2
        x1 = self.branch1(combined)
        x2 = self.branch2(combined)
        x3 = self.branch3(combined)

        combined_branch = torch.cat((x1, x2, x3), dim=1)
        combined_branch = self.concat(combined_branch)
        combined_branch = torch.relu(combined_branch)

        # Classification
        x = torch.relu(self.fc2(combined_branch.flatten(1)))
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output: torch.Size([1, 10])