import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First Block: Three groups with depthwise separable convolutions
        self.group1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0, groups=3),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU(inplace=True)
        )

        self.group2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, groups=3),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU(inplace=True)
        )

        self.group3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2, groups=3),
            nn.BatchNorm2d(num_features=3),
            nn.ReLU(inplace=True)
        )

        # Concatenate the outputs from the three groups
        self.concat1 = nn.Sequential()

        # Second Block: Multiple branches for feature extraction
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels=9, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True)
        )

        # Concatenate the outputs from all branches
        self.concat2 = nn.Sequential()

        # Fully connected layers for classification
        self.fc1 = nn.Linear(in_features=32 * 4 * 4, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # First Block
        x1 = self.group1(x)
        x2 = self.group2(x)
        x3 = self.group3(x)

        # Concatenate the outputs from the three groups
        x = torch.cat([x1, x2, x3], dim=1)

        # Second Block
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)

        # Concatenate the outputs from all branches
        x = torch.cat([branch1_out, branch2_out, branch3_out], dim=1)

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(32, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)