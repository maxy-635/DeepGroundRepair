import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path with two feature extraction paths
        self.main_path = nn.Sequential(
            # First path: 1x1 convolution
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True)
            ),
            # Second path: 1x1, 1x7, 7x1 convolutions
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3)),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0)),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True)
            ),
            # Concatenate and 1x1 convolution to align dimensions
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 1))
        )

        # Branch path (identity)
        self.branch = nn.Identity()

        # Fully connected layers for classification
        self.fc1 = nn.Linear(in_features=32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path
        main_out = self.main_path(x)

        # Branch path
        branch_out = self.branch(x)

        # Add main path and branch outputs
        combined = main_out + branch_out

        # Flatten for fully connected layers
        combined = combined.view(combined.size(0), -1)

        # Fully connected layers
        out = self.fc1(combined)
        out = self.fc2(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(model)