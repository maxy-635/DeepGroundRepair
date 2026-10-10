import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )

        # Branch 1: Local feature extraction
        self.branch1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )

        # Branch 2: Downsample and upsample
        self.branch2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0)
        )

        # Branch 3: Downsample and upsample
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0)
        )

        # Concatenate and 1x1 conv
        self.concat_conv = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )

        # Final fusion and classification
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )
        self.fc = nn.Linear(64 * 32 * 32, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path
        main_out = self.main_path(x)  # output tensor shape: (1, 64, 32, 32)

        # Branch 1
        branch1_out = self.branch1(main_out)  # output tensor shape: (1, 64, 32, 32)

        # Branch 2
        branch2_out = self.branch2(main_out)  # output tensor shape: (1, 64, 32, 32)

        # Branch 3
        branch3_out = self.branch3(main_out)  # output tensor shape: (1, 64, 32, 32)

        # Concatenate branches
        concat_out = torch.cat([branch1_out, branch2_out, branch3_out], dim=1)  # output tensor shape: (1, 192, 32, 32)
        concat_out = self.concat_conv(concat_out)  # output tensor shape: (1, 64, 32, 32)

        # Branch path
        branch_path_out = self.branch_path(x)  # output tensor shape: (1, 64, 32, 32)

        # Fusion
        fusion_out = concat_out + branch_path_out  # output tensor shape: (1, 64, 32, 32)
        fusion_out = self.final_conv(fusion_out)

        # Flatten and classify
        fusion_out = fusion_out.view(fusion_out.size(0), -1)
        out = self.fc(fusion_out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)