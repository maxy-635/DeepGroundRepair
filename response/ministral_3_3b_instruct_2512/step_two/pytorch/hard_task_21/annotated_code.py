import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path with three groups of depthwise separable convolutions
        self.main_path = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True)
        )

        # Concatenate the outputs of the three groups
        self.concat = nn.Sequential(
            nn.Conv2d(32 * 3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Branch path with a 1x1 convolution
        self.branch_path = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Add the outputs of main and branch paths
        self.add = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        # Main path processing
        main_path_output = self.main_path(x)
        # Concatenate outputs from the three groups
        main_path_output = torch.cat([main_path_output, main_path_output, main_path_output], dim=1)
        main_path_output = self.concat(main_path_output)

        # Branch path processing
        branch_path_output = self.branch_path(x)

        # Add outputs from main and branch paths
        combined_output = main_path_output + branch_path_output
        combined_output = self.add(combined_output)

        # Classification
        output = self.fc(combined_output)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor with batch size 1, 3 channels, 32x32 spatial dimensions
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])