import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path with multi-scale feature extraction
        self.main_path = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(num_features=64),
                nn.ReLU(inplace=True)
            )
        ])

        # Branch path with 1x1 convolution
        self.branch_path = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # Fully connected layers for classification
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 32 * 32, out_features=512),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        # Process the entire input through each main path module
        out1 = self.main_path[0](x)
        out2 = self.main_path[1](x)
        out3 = self.main_path[2](x)

        # Concatenate the outputs
        main_out = torch.cat([out1, out2, out3], dim=1)

        # Process input through the branch path
        branch_out = self.branch_path(x)

        # Fuse the outputs
        fused_out = main_out + branch_out

        # Pass through fully connected layers
        output = self.fc_layers(fused_out)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # CIFAR-10 image shape: (3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor)