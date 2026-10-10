import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Primary path
        self.conv1x1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.depthwise_conv_3x3_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32)
        self.conv1x1_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Block 1: Branch path
        self.depthwise_conv_3x3_2 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, groups=1)
        self.conv1x1_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Block 2: Channel shuffling
        self.fc = nn.Linear(in_features=64*28*28, out_features=10)  # Assuming input size is 28x28 and 10 classes

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Block 1: Primary path
        x_primary = self.conv1x1_1(x)  # output tensor shape: (1, 32, 28, 28)
        x_primary = self.depthwise_conv_3x3_1(x_primary)  # output tensor shape: (1, 32, 28, 28)
        x_primary = self.conv1x1_2(x_primary)  # output tensor shape: (1, 64, 28, 28)

        # Block 1: Branch path
        x_branch = self.depthwise_conv_3x3_2(x)  # output tensor shape: (1, 32, 28, 28)
        x_branch = self.conv1x1_3(x_branch)  # output tensor shape: (1, 64, 28, 28)

        # Concatenate along channel dimension
        x = torch.cat([x_primary, x_branch], dim=1)  # output tensor shape: (1, 128, 28, 28)

        # Block 2: Channel shuffling
        _, _, h, w = x.shape  # output tensor shape: (1); (1); (1)
        x = x.view(h, w, -1)  # output tensor shape: (28, 28, 128)
        x = x.permute(2, 0, 1)  # output tensor shape: (128, 28, 28)
        x = x.view(1, -1, h, w)  # output tensor shape: (1, 128, 28, 28)

        # Flatten and pass through fully connected layer
        x = x.view(1, -1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Dummy input tensor for MNIST (batch_size=1, channels=1, height=28, width=28)
    output_tensor = model(input_tensor)
    print(output_tensor)