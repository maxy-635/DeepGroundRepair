import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=0.2)
        )

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.2)
        )

        # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.2)
        )

        # Branch 4: Average pooling followed by 1x1 convolution
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=32, stride=2, padding=31),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Dropout(p=0.2)
        )

        # Concatenate outputs from all branches
        self.concat_layer = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
        )

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=256 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Pass input through each branch
        branch1_out = self.branch1(x)  # output tensor shape: (1, 64, 32, 32)
        branch2_out = self.branch2(x)  # output tensor shape: (1, 64, 32, 32)
        branch3_out = self.branch3(x)  # output tensor shape: (1, 64, 32, 32)
        branch4_out = self.branch4(x)  # output tensor shape of this layer should be (1, 64, 32, 32)

        # Concatenate outputs
        concat_out = torch.cat((branch1_out, branch2_out, branch3_out, branch4_out), dim=1)
        concat_out = self.concat_layer(concat_out)

        # Flatten the output
        flattened = torch.flatten(concat_out, 1)

        # Pass through fully connected layers
        fc1_out = torch.relu(self.fc1(flattened))
        fc2_out = torch.relu(self.fc2(fc1_out))
        fc3_out = self.fc3(fc2_out)

        return fc3_out


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (CIFAR-10 images are 32x32)
    input_tensor = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # Expected: torch.Size([1, 10])