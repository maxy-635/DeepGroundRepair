import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First feature extraction path: 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64)
        )

        # Second feature extraction path: 1x1, 1x7, 7x1 convolutions
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64)
        )

        # Concatenate and align dimensions with 1x1 convolution
        self.concat_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64)
        )

        # Branch that connects directly to the input
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64)
        )

        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(in_features=64 * 32 * 32, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        # Main path
        path1_out = self.path1(x)
        path2_out = self.path2(x)
        concat_out = torch.cat([path1_out, path2_out], dim=1)
        main_out = self.concat_conv(concat_out)

        # Branch path
        branch_out = self.branch(x)

        # Merge main and branch outputs
        merged_out = main_out + branch_out

        # Flatten and pass through fully connected layers
        flattened = merged_out.view(merged_out.size(0), -1)
        output = self.fc(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)