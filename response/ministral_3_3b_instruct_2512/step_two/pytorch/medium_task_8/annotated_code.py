import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask'),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        # Concatenation and final layers
        self.final_conv = nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Split the input into three groups along the last dimension
        x1, x2, x3 = x.chunk(3, dim=1)  # output tensor shape: (1, 1, 32, 32); (1, 1, 32, 32); (1, 1, 32, 32)

        # Process the second group in the main path
        x2_processed = self.main_path(x2)

        # Combine the second and third groups
        combined = torch.cat([x2_processed, x3], dim=1)

        # Process the combined output
        main_path_output = self.final_conv(combined)

        # Process the branch path
        branch_path_output = self.branch_path(x1)

        # Add the outputs from main and branch paths
        fused_output = main_path_output + branch_path_output

        # Flatten and pass through the fully connected layer
        output = self.flatten(fused_output)
        output = self.fc(output)

        return output

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, 3, 32, 32)
    input_tensor = torch.randn(1, 3, 32, 32)
    model = DL_Model()
    output_tensor = model(input_tensor)
    print(output_tensor.shape)