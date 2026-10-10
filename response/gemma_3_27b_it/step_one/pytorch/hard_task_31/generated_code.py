import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Main and Branch Paths
        self.block1_main = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        )

        # Block 2: Separable Convolution Groups
        self.block2_group1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=0.2)
        )
        self.block2_group2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.2)
        )
        self.block2_group3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.Dropout(p=0.2)
        )

        # Final Layers
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=32 * 32 * 32, out_features=10)  # Assuming 32x32 input image

    def forward(self, x):
        # Block 1
        main_path_output = self.block1_main(x)
        branch_path_output = x
        block1_output = main_path_output + branch_path_output

        # Block 2
        group1_output = self.block2_group1(block1_output)
        group2_output = self.block2_group2(block1_output)
        group3_output = self.block2_group3(block1_output)
        block2_output = torch.cat((group1_output, group2_output, group3_output), dim=1)

        # Final Layers
        flattened_output = self.flatten(block2_output)
        output = self.linear(flattened_output)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)