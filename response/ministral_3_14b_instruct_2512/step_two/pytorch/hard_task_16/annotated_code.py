import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Split into three groups with 1x1, 3x3, 1x1 convolutions
        self.block1_conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )

        self.block1_conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )

        self.block1_conv3 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(3)
        )

        # Transition Convolution
        self.transition_conv = nn.Sequential(
            nn.Conv2d(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )

        # Block 2: Global max pooling, FC layers, and weight generation
        self.global_max_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True)
        )

        # Final classification layer
        self.fc_classifier = nn.Linear(64, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1: Split into three groups
        x1 = self.block1_conv1(x)  # output tensor shape: (1, 64, 32, 32)
        x2 = self.block1_conv2(x1)  # output tensor shape: (1, 64, 32, 32)
        x3 = self.block1_conv3(x2)  # output tensor shape: (1, 3, 32, 32)

        # Concatenate along channel dimension
        x_concat = torch.cat([x1, x2, x3], dim=1)  # output tensor shape: (1, 131, 32, 32)

        # Transition Convolution
        x_transition = self.transition_conv(x_concat)

        # Block 2: Global max pooling
        x_pool = self.global_max_pool(x_transcat)

        # Flatten for FC layers
        x_flat = x_pool.view(x_pool.size(0), -1)

        # Generate weights through FC layers
        weights = self.fc1(x_flat)
        weights = self.fc2(weights)

        # Reshape weights to match the shape of x_transition
        weights = weights.view(weights.size(0), 64, 1, 1)

        # Multiply weights with x_transition
        x_main = x_transition * weights

        # Branch: Direct connection to input
        x_branch = x

        # Add main path and branch
        x_add = x_main + x_branch

        # Final classification
        x_flat_final = x_add.view(x_add.size(0), -1)
        x_out = self.fc_classifier(x_flat_final)

        return x_out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)