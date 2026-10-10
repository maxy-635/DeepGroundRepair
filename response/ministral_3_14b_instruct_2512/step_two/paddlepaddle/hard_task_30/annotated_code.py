import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DualPathBlock(nn.Layer):
    def __init__(self, in_channels):
        super(DualPathBlock, self).__init__()
        self.main_path = nn.Sequential(
            nn.Conv2D(in_channels=in_channels, out_channels=in_channels*2,
                     kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2D(in_channels=in_channels*2, out_channels=in_channels,
                     kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.branch_path = nn.Identity()

    def forward(self, x):
        main_out = self.main_path(x)
        branch_out = self.branch_path(x)
        return main_out + branch_out

class ChannelSplitBlock(nn.Layer):
    def __init__(self, in_channels):
        super(ChannelSplitBlock, self).__init__()
        self.split_channels = in_channels // 3
        self.conv1x1 = nn.Sequential(
            nn.Conv2D(in_channels=self.split_channels, out_channels=self.split_channels,
                     kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.conv3x3 = nn.Sequential(
            nn.Conv2D(in_channels=self.split_channels, out_channels=self.split_channels,
                     kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv5x5 = nn.Sequential(
            nn.Conv2D(in_channels=self.split_channels, out_channels=self.split_channels,
                     kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

    def forward(self, x):
        # Split channels into three groups
        x1, x2, x3 = paddle.split(x, [self.split_channels]*3, axis=1)

        # Process each group with different kernel sizes
        x1 = self.conv1x1(x1)
        x2 = self.conv3x3(x2)
        x3 = self.conv5x5(x3)

        # Concatenate along channel dimension
        return paddle.concat([x1, x2, x3], axis=1)

class DL_Model(nn.Layer):
    def __init__(self):
        super(DualPathBlock, self).__init__()
        self.dual_path_block = DualPathBlock(in_channels=3)
        self.channel_split_block = ChannelSplitBlock(in_channels=3)
        self.avg_pool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc1 = nn.Linear(in_features=3*3*3, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # First block - Dual path
        x = self.dual_path_block(x)

        # Second block - Channel split with different kernel sizes
        x = self.channel_split_block(x)

        # Global average pooling
        x = self.avg_pool(x)
        x = paddle.flatten(x, start_axis=1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # CIFAR-10 image shape
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)