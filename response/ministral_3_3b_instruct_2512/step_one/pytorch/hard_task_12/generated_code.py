import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),  # Dimensionality reduction
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0),  # 1x1 convolution
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # 3x3 convolution
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)  # Matching dimensions
        )

        # Concatenate and add layers
        self.concatenate = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),  # Concatenate outputs
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),  # Assuming output size after flattening is 8x8
            nn.Linear(128, 10)  # Assuming 10 classes
        )

    def forward(self, x):
        main_output = self.main_path(x)
        branch_output = self.branch_path(x)

        # Concatenate outputs from main and branch paths
        concatenated = torch.cat((main_output, branch_output), dim=1)

        # Add outputs from main and branch paths
        added_output = main_output + branch_output

        # Use concatenated output for classification
        output = self.concatenate(concatenated)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 64, 32, 32)  # Dummy input tensor with shape (1, 64, 32, 32)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)