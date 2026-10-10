import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Define the block that will be used in both branches
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(in_features='mask', out_features=32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU()
        )

        # Branch 1
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU()
        )

        # Branch 2
        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU()
        )

        # Final layers
        self.concatenate = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.GELU()
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Process both branches
        branch1_output = self.branch1(x)  # output tensor shape: (1, 32)
        branch2_output = self.branch2(x)  # output tensor shape: (1, 32)

        # Apply the block to both branch outputs
        branch1_output = self.block(branch1_output)
        branch2_output = self.block(branch2_output)

        # Element-wise multiplication
        branch1_output = branch1_output * branch1_output
        branch2_output = branch2_output * branch2_output

        # Concatenate outputs
        concatenated = torch.cat([branch1_output, branch2_output], dim=1)

        # Final classification layer
        output = self.concatenate(concatenated)
        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    # For CIFAR-10, height and width are 32, channels are 3
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)