import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.conv1 = nn.Conv2d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Branch Path
        self.branch_conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Fully Connected Layer
        self.fc = nn.Linear(in_features=32 * 32 * 32, out_features=10)  # Assuming input size is 32x32
        self.flatten = nn.Flatten()

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main Path
        group1, group2, group3 = torch.split(x, 1, dim=1)  # output tensor shape: (1, 1, 32, 32); (1, 1, 32, 32); (1, 1, 32, 32)

        group2_processed = self.conv1(group2)
        combined_group = torch.add(group2_processed, group3)
        combined_group_processed = self.conv2(combined_group)

        main_path_output = torch.concat([group1, combined_group_processed], dim=1)

        # Branch Path
        branch_path_output = self.branch_conv(x)

        # Fusion
        fused_output = torch.add(main_path_output, branch_path_output)

        # Classification
        flattened_output = self.flatten(fused_output)
        output = self.fc(flattened_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)