import torch
import torch.nn as nn

class DL_Model(nn.Module):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.initial_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Block 1
        self.block1_conv1x1_1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.block1_depthwise_conv = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1, groups=16)
        self.block1_conv1x1_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Block 2
        self.block2_shuffle = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 10)
        )

        # Channel shuffling implementation
        self.channel_shuffle = nn.Sequential(
            nn.Lambda(lambda x: x.view(x.size(1), x.size(2), x.size(3), x.size(4) // 4, 4)),
            nn.Lambda(lambda x: x.permute(0, 1, 2, 4, 3)),
            nn.Lambda(lambda x: x.view(x.size(0), x.size(1), x.size(2), x.size(3) * x.size(4))),
        )

    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)

        # Block 1
        # Split into two groups
        x1, x2 = x.chunk(2, dim=1)

        # Process first group
        x1 = self.block1_conv1x1_1(x1)
        x1 = self.block1_depthwise_conv(x1)
        x1 = self.block1_conv1x1_2(x1)

        # Concatenate
        x = torch.cat([x1, x2], dim=1)

        # Block 2
        # Channel shuffling
        x = self.channel_shuffle(x)

        # Flatten and classification
        x = x.view(x.size(0), -1)
        x = self.block2_shuffle(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = torch.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)