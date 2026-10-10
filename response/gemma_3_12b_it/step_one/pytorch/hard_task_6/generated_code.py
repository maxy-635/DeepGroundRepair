import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_split = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.block1_relu = nn.ReLU()
        self.block1_concat = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Block 2
        self.block2_shape = nn.Identity() # Placeholder, shape information is used for reshaping
        self.block2_permute = nn.Identity() # Placeholder, permutation is used for channel shuffling
        self.block2_reshape = nn.Identity() # Placeholder, reshaping is used for channel shuffling

        # Block 3
        self.block3_depthwise = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, groups=1, bias=False)
        self.block3_relu = nn.ReLU()

        # Branch path
        self.branch_avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.branch_flatten = nn.Flatten() # Flatten the output of AvgPool2d

        # Final layers
        self.fc = nn.Linear(in_features=1024, out_features=10)

    def forward(self, x):
        # Block 1
        x1 = self.block1_split(x)
        x1 = self.block1_relu(x1)
        x2 = self.block1_split(x)
        x2 = self.block1_relu(x2)
        x3 = self.block1_split(x)
        x3 = self.block1_relu(x3)
        x_block1 = torch.cat([x1, x2, x3], dim=1)
        x_block1 = self.block1_concat(x_block1)

        # Block 2
        shape = x_block1.shape
        groups = 3
        channels_per_group = shape[1] // groups
        x_reshaped = x_block1.reshape(shape[0], shape[1], groups, channels_per_group)
        x_permuted = x_reshaped.permute(0, 2, 3, 1)
        x_block2 = x_permuted.reshape(shape[0], shape[1], groups * channels_per_group)

        # Block 3
        x_depthwise = self.block3_depthwise(x_block2)
        x_block3 = self.block3_relu(x_depthwise)

        # Branch path
        x_branch = self.branch_avgpool(x)
        x_branch = self.branch_flatten(x_branch)

        # Concatenate and final layers
        x_combined = torch.cat([x_block3, x_branch], dim=1)
        x_final = torch.relu(x_combined)
        x_output = self.fc(x_final)

        return x_output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)