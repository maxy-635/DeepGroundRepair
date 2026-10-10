import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(DL_Model, self).__init__()

        # Main Path
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(1, 64), # Apply GroupNorm for better regularization
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, 64),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(1, 64),
        )

        self.branch1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, 64)
        )

        self.branch2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )

        self.concat_conv = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(1, 64)
        )

        # Branch Path
        self.branch_path = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(1, 64)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Main Path
        main_path_output = self.main_path(x)
        branch1_output = self.branch1(main_path_output)
        branch2_output = self.branch2(main_path_output)
        branch3_output = self.branch3(main_path_output)

        combined_output = torch.cat((branch1_output, branch2_output, branch3_output), dim=1)
        main_path_final = self.concat_conv(combined_output)

        # Branch Path
        branch_path_output = self.branch_path(x)

        # Combine Paths
        combined = main_path_final + branch_path_output

        # Classifier
        flattened = combined.view(combined.size(0), -1)
        output = self.classifier(flattened)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)