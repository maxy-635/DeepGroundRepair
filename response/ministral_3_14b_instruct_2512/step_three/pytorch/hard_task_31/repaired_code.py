import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block with main and branch paths
        self.first_block = nn.Sequential(
            # Main path
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
                nn.Dropout2d(p=0.2),
                nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0)
            ),
            # Branch path (identity connection)
            nn.Identity()
        )

        # Second block with three separable convolution groups
        self.second_block = nn.Sequential(
            # Group 1: 1x1 separable convolution
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
                nn.Dropout2d(p=0.2)
            ),
            # Group 2: 3x3 separable convolution
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.Dropout2d(p=0.2)
            ),
            # Group 3: 5x5 separable convolution
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
                nn.Dropout2d(p=0.2)
            )
        )

        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=96*32*32, out_features=10)
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First block processing
        main_path = self.first_block[0](x)  # output tensor shape: (1, 3, 32, 32)
        branch_path = self.first_block[1](x)  # output tensor shape: (1, 3, 32, 32)
        block1_output = main_path + branch_path  # output tensor shape: (1, 3, 32, 32)

        # Second block processing
        # Split input into three groups along channel dimension
        group1 = self.second_block[0](block1_output)  # output tensor shape: (1, 32, 32, 32)
        group2 = self.second_block[1](block1_output)  # output tensor shape: (1, 32, 32, 32)
        group3 = self.second_block[2](block1_output)  # output tensor shape: (1, 32, 32, 32)

        # Concatenate along channel dimension
        block2_output = torch.cat([group1, group2, group3], dim=1)  # output tensor shape: (1, 96, 32, 32)

        # Final classification
        output = self.classifier(block2_output)
        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)