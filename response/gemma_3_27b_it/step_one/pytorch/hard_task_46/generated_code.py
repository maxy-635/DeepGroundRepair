import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Separable Convolution with different kernel sizes
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv1_5 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

        # Block 2: Multiple branches for enhanced feature extraction
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.branch3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Fully Connected Layer
        self.fc = nn.Linear(in_features=64*1*1, out_features=10)  # Adjust input features based on concatenation

    def forward(self, x):
        # Block 1
        out1_1 = self.conv1_1(x)
        out1_3 = self.conv1_3(x)
        out1_5 = self.conv1_5(x)
        x = torch.cat((out1_1, out1_3, out1_5), dim=1)  # Concatenate along the channel dimension

        # Block 2
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)

        x = torch.cat((branch1_out, branch2_out, branch3_out), dim=1)

        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  # Flatten the tensor

        # Fully Connected Layer
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)