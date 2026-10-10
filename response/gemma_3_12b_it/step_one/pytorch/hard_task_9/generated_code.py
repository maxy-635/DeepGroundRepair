import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Feature extraction branches
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Concatenation and adjustment
        self.concat_conv = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Direct connection branch
        self.direct_branch = nn.Sequential(
            nn.BatchNorm2d(3)
        )

        # Fusion
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # Feature extraction
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)

        # Concatenation
        concat_out = torch.cat((branch1_out, branch2_out, branch3_out), dim=1)

        # Adjustment
        adjusted_out = self.concat_conv(concat_out)

        # Direct connection
        direct_out = self.direct_branch(x)

        # Fusion
        fused_out = torch.add(adjusted_out, direct_out)

        # Flatten
        flattened_out = torch.flatten(fused_out, 1)

        # Fully connected layers
        fc1_out = torch.relu(self.fc1(flattened_out))
        fc2_out = torch.relu(self.fc2(fc1_out))
        fc3_out = self.fc3(fc2_out)

        return fc3_out


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)