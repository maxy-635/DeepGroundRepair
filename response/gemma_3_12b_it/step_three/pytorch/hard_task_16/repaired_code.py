import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_group1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0)
        )
        self.block1_group2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0)
        )
        self.block1_group3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0)
        )
        self.block1_concat = nn.Conv2d(in_channels=16*3, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Transition Convolution
        self.transition_conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Block 2
        self.block2_global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.block2_fc1 = nn.Linear(in_features=64, out_features=32)
        self.block2_fc2 = nn.Linear(in_features=32, out_features=64)

        # Direct Connection Branch
        self.direct_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Final Classification
        self.final_fc = nn.Linear(in_features=64 + 64, out_features=10)


    def forward(self, x):
        # Main Path
        x1 = self.block1_group1(x)
        x2 = self.block1_group2(x)
        x3 = self.block1_group3(x)
        x_concat = torch.cat((x1, x2, x3), dim=1)
        x_block1 = self.block1_concat(x_concat)
        x_transition = self.transition_conv(x_block1)

        x_global_pool = self.block2_global_pool(x_transition)
        x_global_pool = x_global_pool.view(-1, 64)
        x_weights = self.block2_fc2(F.relu(self.block2_fc1(x_global_pool)))
        x_weights = x_weights.view(-1, 64)
        x_main_path = x_transition * x_weights

        # Branch
        x_branch = self.direct_conv(x)

        # Add and Classify
        x_combined = torch.cat((x_main_path, x_branch), dim=1)
        x_output = self.final_fc(x_combined)

        return x_output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: torch.Size([1, 10])