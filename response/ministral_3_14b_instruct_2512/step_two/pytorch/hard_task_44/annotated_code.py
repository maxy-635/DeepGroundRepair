import torch
import torch.nn as nn
import torch.nn.functional as F

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three parallel convolutional branches with different kernel sizes
        self.conv1x1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)

        self.dropout = nn.Dropout(p=0.5)

        # Block 2: Four branches with different operations
        # Branch 1: 1x1 conv
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Branch 2: 1x1 conv + 3x3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        )

        # Branch 3: 1x1 conv + 5x5 conv
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        )

        # Branch 4: 3x3 max pooling + 1x1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features='mask', out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1 processing
        x1 = self.conv1x1(x)  # output tensor shape: (1, 64, 32, 32)
        x3 = self.conv3x3(x)  # output tensor shape: (1, 64, 32, 32)
        x5 = self.conv5x5(x)  # output tensor shape: (1, 64, 32, 32)

        # Concatenate along channel dimension
        x = torch.cat([x1, x3, x5], dim=1)  # output tensor shape: (1, 192, 32, 32)
        x = self.dropout(x)  # output tensor shape: (1, 192, 32, 32)

        # Block 2 processing
        branch1_out = self.branch1(x)  # output tensor shape: (1, 64, 32, 32)
        branch2_out = self.branch2(x)  # output tensor shape: (1, 64, 32, 32)
        branch3_out = self.branch3(x)  # output tensor shape: (1, 64, 32, 32)
        branch4_out = self.branch4(x)  # output tensor shape: (1, 64, 32, 32)

        # Concatenate all branches
        x = torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], dim=1)  # output tensor shape: (1, 256, 32, 32)

        # Final layers
        x = self.flatten(x)  # output tensor shape: (1, 262144)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = torch.randn(1, 3, 32, 32)  # batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)