import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: main path and branch path
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.5),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        )
        self.branch_path = nn.Identity()  # Direct connection

        # Second block: separable convolutions with different kernel sizes
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=0.5)
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.5)
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.Dropout(p=0.5)
        )

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=32 * 32 * 96, out_features=10)  # 96 channels from concatenation (32*3)

    def forward(self, x):
        # First block
        main_out = self.main_path(x)
        branch_out = self.branch_path(x)
        block1_out = main_out + branch_out

        # Second block
        x1 = self.conv1x1(block1_out)
        x3 = self.conv3x3(block1_out)
        x5 = self.conv5x5(block1_out)
        block2_out = torch.cat([x1, x3, x5], dim=1)

        # Final layers
        out = self.flatten(block2_out)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # CIFAR-10 input shape: (batch_size, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor)