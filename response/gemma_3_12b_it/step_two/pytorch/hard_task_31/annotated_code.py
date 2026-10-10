import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First Block
        self.block1_main_path = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        )
        self.block1_branch_path = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        )

        # Second Block
        self.block2_conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.block2_dropout1 = nn.Dropout(p=0.2)
        self.block2_conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.block2_dropout2 = nn.Dropout(p=0.2)
        self.block2_conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.block2_dropout3 = nn.Dropout(p=0.2)

        # Final Layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features='mask', out_features=10)


    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First Block
        main_path_output = self.block1_main_path(x)  # output tensor shape: (1, 3, 32, 32)
        branch_path_output = self.block1_branch_path(x)  # output tensor shape: (1, 3, 32, 32)
        block1_output = main_path_output + branch_path_output  # output tensor shape: (1, 3, 32, 32)

        # Second Block
        group1 = block1_output.split(1, dim=3)  # output tensor shape: (unknown)
        group2 = block1_output.split(1, dim=3)  # output tensor shape: (unknown)
        group3 = block1_output.split(1, dim=3)  # output tensor shape: (unknown)

        group1_output = self.block2_conv1(group1[0])  # output tensor shape: (1, 32, 32, 1)
        group1_output = self.block2_dropout1(group1_output)  # output tensor shape: (1, 32, 32, 1)
        group1_output = self.block2_conv2(group1_output)  # output tensor shape: (1, 32, 32, 1)
        group1_output = self.block2_dropout2(group1_output)  # output tensor shape: (1, 32, 32, 1)
        group1_output = self.block2_conv3(group1_output)  # output tensor shape: (1, 32, 32, 1)
        group1_output = self.block2_dropout3(group1_output)  # output tensor shape: (1, 32, 32, 1)


        group2_output = self.block2_conv1(group2[0])  # output tensor shape: (1, 32, 32, 1)
        group2_output = self.block2_dropout1(group2_output)  # output tensor shape: (1, 32, 32, 1)
        group2_output = self.block2_conv2(group2_output)  # output tensor shape: (1, 32, 32, 1)
        group2_output = self.block2_dropout2(group2_output)  # output tensor shape: (1, 32, 32, 1)
        group2_output = self.block2_conv3(group2_output)  # output tensor shape: (1, 32, 32, 1)
        group2_output = self.block2_dropout3(group2_output)  # output tensor shape: (1, 32, 32, 1)

        group3_output = self.block2_conv1(group3[0])  # output tensor shape: (1, 32, 32, 1)
        group3_output = self.block2_dropout1(group3_output)  # output tensor shape: (1, 32, 32, 1)
        group3_output = self.block2_conv2(group3_output)  # output tensor shape: (1, 32, 32, 1)
        group3_output = self.block2_dropout2(group3_output)  # output tensor shape: (1, 32, 32, 1)
        group3_output = self.block2_conv3(group3_output)  # output tensor shape: (1, 32, 32, 1)
        group3_output = self.block2_dropout3(group3_output)  # output tensor shape: (1, 32, 32, 1)

        block2_output = torch.cat((group1_output, group2_output, group3_output), dim=3)  # output tensor shape: (1, 32, 32, 3)

        # Final Layers
        x = self.flatten(block2_output)  # output tensor shape: (1, 3072)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)