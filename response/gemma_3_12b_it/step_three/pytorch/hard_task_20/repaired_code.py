import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.main_path = nn.Sequential(
            # Group 1: 1x1 kernel
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(1, 1), stride=1, padding=0),
                nn.ReLU(inplace=True)
            ),
            # Group 2: 3x3 kernel
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU(inplace=True)
            ),
            # Group 3: 5x5 kernel
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=1, padding=2),
                nn.ReLU(inplace=True)
            )
        )
        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=192, kernel_size=(1, 1), stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path
        main_path_output = self.main_path[0](x)  # output tensor shape: (1, 64, 32, 32)
        main_path_output = torch.cat([main_path_output, self.main_path[1](x), self.main_path[2](x)], dim=1)  # output tensor shape: (1, 192, 32, 32)

        # Branch path
        branch_path_output = self.branch_path(x)  # output tensor shape of this layer should be (1, 192, 32, 32)

        # Fuse features
        fused_features = torch.add(main_path_output, branch_path_output)

        # Flatten for fully connected layers
        fused_features = fused_features.view(fused_features.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(fused_features))
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for CIFAR-10 (3 channels, 32x32)
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)