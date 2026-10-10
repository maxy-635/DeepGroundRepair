import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Dual-path structure
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=1)
        )

        # Second block: Multi-group depthwise separable convolutions
        self.group1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, groups=1),
            nn.ReLU(inplace=True)
        )

        self.group3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, groups=1),
            nn.ReLU(inplace=True)
        )

        self.group5 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2, groups=1),
            nn.ReLU(inplace=True)
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(in_features=32*32*3, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        # First block
        main_out = self.main_path(x)
        branch_out = x
        out1 = main_out + branch_out

        # Second block
        # Split into three groups along channel dimension
        x1, x2, x3 = torch.chunk(out1, 3, dim=1)

        out2_1 = self.group1(x1)
        out2_2 = self.group3(x2)
        out2_3 = self.group5(x3)

        out2 = torch.cat([out2_1, out2_2, out2_3], dim=1)

        # Flatten and pass through fully connected layers
        out = out2.view(out2.size(0), -1)
        out = self.fc(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # CIFAR-10 image shape: (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)