import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class ChannelAttentionBlock(nn.Layer):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)

        # Path1: Global average pooling + 2 FC layers
        self.fc1_avg = nn.Linear(in_channels, in_channels // 16)
        self.fc2_avg = nn.Linear(in_channels // 16, in_channels)

        # Path2: Global max pooling + 2 FC layers
        self.fc1_max = nn.Linear(in_channels, in_channels // 16)
        self.fc2_max = nn.Linear(in_channels // 16, in_channels)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Path1
        avg_out = self.avg_pool(x)
        avg_out = avg_out.reshape([avg_out.shape[0], -1])
        avg_out = self.fc1_avg(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2_avg(avg_out)

        # Path2
        max_out = self.max_pool(x)
        max_out = max_out.reshape([max_out.shape[0], -1])
        max_out = self.fc1_max(max_out)
        max_out = self.relu(max_out)
        max_out = self.fc2_max(max_out)

        # Combine paths
        out = avg_out + max_out
        out = self.sigmoid(out)
        out = out.reshape([-1, x.shape[1], 1, 1])

        return x * out

class SpatialAttentionBlock(nn.Layer):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__()
        self.conv = nn.Conv2D(in_channels=2, out_channels=1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = F.adaptive_avg_pool2d(x, 1)
        max_out = F.adaptive_max_pool2d(x, 1)

        out = paddle.concat([avg_out, max_out], axis=1)
        out = self.conv(out)
        out = self.sigmoid(out)

        return x * out

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, padding=1)

        # Channel attention block
        self.ca_block = ChannelAttentionBlock(in_channels=64)

        # Spatial attention block
        self.sa_block = SpatialAttentionBlock(in_channels=64)

        # Additional branch for channel alignment
        self.conv_branch = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1)

        # Final classification layer
        self.fc = nn.Linear(64 * 32 * 32, 10)  # CIFAR-10 has 10 classes

        # Activation functions
        self.relu = nn.ReLU()

    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.relu(x)

        # Channel attention
        x = self.ca_block(x)

        # Spatial attention
        x = self.sa_block(x)

        # Additional branch
        branch = self.conv_branch(x)
        x = x + branch
        x = self.relu(x)

        # Flatten and classify
        x = x.flatten(start_axis=1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([32, 3, 32, 32])  # Batch of 32 CIFAR-10 images (3 channels, 32x32)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # Should be [32, 10]