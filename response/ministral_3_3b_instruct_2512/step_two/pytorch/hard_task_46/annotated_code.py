import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Separable Convolutional Layers
        self.separable_conv_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)  # Concatenating outputs
        )

        # Second block: Multi-branch Feature Extraction
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask'),
            nn.BatchNorm2d(128)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )

        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.concatenate_block = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256)
        )

        # Final layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First block
        x1 = self.separable_conv_block[0](x)  # output tensor shape: (1, 64, 32, 32)
        x2 = self.separable_conv_block[1](x1)  # output tensor shape: (1, 64, 32, 32)
        x3 = self.separable_conv_block[2](x2)  # output tensor shape: (1, 64, 32, 32)
        x4 = self.separable_conv_block[3](x3)  # output tensor shape: (1, 128, 32, 32)

        # Concatenate outputs from the first block
        x_concat = torch.cat((x1, x2, x3), dim=1)  # output tensor shape: (1, 192, 32, 32)

        # Second block
        branch1_out = self.branch1(x_concat)
        branch2_out = self.branch2(x_concat)
        branch3_out = self.branch3(x_concat)

        # Concatenate outputs from the branches
        x_branch_concat = torch.cat((branch1_out, branch2_out, branch3_out), dim=1)

        # Further processing
        x_branch_concat = self.concatenate_block(x_branch_concat)

        # Global average pooling and fully connected layer
        x_pool = self.global_avg_pool(x_branch_concat)
        x_pool = torch.flatten(x_pool, 1)
        output = self.fc(x_pool)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Example input tensor for CIFAR-10 (batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)