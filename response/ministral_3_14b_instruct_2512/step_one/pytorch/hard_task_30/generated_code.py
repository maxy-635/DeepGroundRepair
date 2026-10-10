import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: dual-path structure
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=3)
        )

        self.branch_path = nn.Sequential()

        # Second block: three-group feature extraction
        self.group1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0, groups=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=3)
        )

        self.group2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, groups=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=3)
        )

        self.group3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2, groups=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=3)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=3*32*32, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # First block processing
        main_output = self.main_path(x)
        branch_output = self.branch_path(x)
        block1_output = main_output + branch_output

        # Second block processing
        group1_output = self.group1(block1_output)
        group2_output = self.group2(block1_output)
        group3_output = self.group3(block1_output)

        # Concatenate along channel dimension
        block2_output = torch.cat([group1_output, group2_output, group3_output], dim=1)

        # Flatten for fully connected layers
        block2_output = block2_output.view(block2_output.size(0), -1)

        # Fully connected layers
        fc_output = F.relu(self.fc1(block2_output))
        output = self.fc2(fc_output)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)