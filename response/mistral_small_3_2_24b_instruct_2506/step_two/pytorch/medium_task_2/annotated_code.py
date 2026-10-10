import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size='mask', stride='mask', padding='mask')
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32*16*16, out_features=128),
            nn.Linear(in_features=128, out_features=10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Process through main path
        main_out = self.main_path(x)  # output tensor shape of this layer should be (1, 32, 32, 32)

        # Process through branch path
        branch_out = self.branch_path(x)  # output tensor shape: (1, 32, 32, 32)

        # Combine features from both paths
        combined = torch.cat((main_out, branch_out), dim=1)

        # Pass through fully connected layers
        out = self.fc_layers(combined)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor)