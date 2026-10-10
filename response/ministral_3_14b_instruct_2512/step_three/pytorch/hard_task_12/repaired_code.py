import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0),  # 1x1 conv for dimensionality reduction
            nn.ReLU(inplace=True)
        )

        # Parallel convolutional layers in main path
        self.conv1x1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),  # 3x3 conv to match main path channels
            nn.ReLU(inplace=True)
        )

        # Classification head
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32*32*32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # Assuming 10 classes for classification

    def forward(self, x):
        # Main path processing
        main_out = self.main_path(x)
        conv1x1_out = self.conv1x1(main_out)
        conv3x3_out = self.conv3x3(main_out)
        main_combined = torch.cat([conv1x1_out, conv3x3_out], dim=1)  # Concatenate along channel dimension

        # Branch path processing
        branch_out = self.branch_path(x)

        # Combine main and branch paths
        combined = torch.add(main_combined, branch_out)

        # Classification
        flattened = self.flatten(combined)
        fc1_out = self.fc1(flattened)
        fc2_out = self.fc2(fc1_out)

        return fc2_out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 64, 32, 32)  # Batch size 1, 64 channels, 32x32 spatial dimensions
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)