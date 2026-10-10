import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial 1x1 convolution to increase channels threefold
        self.conv1x1_expand = nn.Conv2D(
            in_channels=3,
            out_channels=9,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Depthwise separable convolution (3x3 depthwise + 1x1 pointwise)
        self.depthwise_conv = nn.Conv2D(
            in_channels=9,
            out_channels=9,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=9  # Depthwise convolution
        )
        self.pointwise_conv = nn.Conv2D(
            in_channels=9,
            out_channels=9,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Channel attention mechanism
        self.global_avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.fc1 = nn.Linear(in_features=9, out_features=9)
        self.fc2 = nn.Linear(in_features=9, out_features=9)

        # Final 1x1 convolution to reduce dimensionality
        self.conv1x1_reduce = nn.Conv2D(
            in_channels=9,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Classification layers
        self.flatten = nn.Flatten()
        self.fc_classifier = nn.Linear(in_features=3*32*32, out_features=10)  # Assuming CIFAR-10 output

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initial feature extraction
        x = self.conv1x1_expand(x)
        x = self.relu(x)

        # Depthwise separable convolution
        x = self.depthwise_conv(x)
        x = self.relu(x)
        x = self.pointwise_conv(x)
        x = self.relu(x)

        # Channel attention
        # Global average pooling
        channel_attention = self.global_avg_pool(x)
        channel_attention = self.flatten(channel_attention)

        # Two fully connected layers
        channel_attention = self.fc1(channel_attention)
        channel_attention = self.relu(channel_attention)
        channel_attention = self.fc2(channel_attention)
        channel_attention = self.sigmoid(channel_attention)

        # Reshape and multiply with original features
        channel_attention = channel_attention.view(-1, 9, 1, 1)
        x = x * channel_attention

        # Final 1x1 convolution
        x = self.conv1x1_reduce(x)
        x = self.relu(x)

        # Add skip connection
        x = x + self.conv1x1_expand(x)  # This might need adjustment based on exact requirements

        # Classification
        x = self.flatten(x)
        x = self.fc_classifier(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch of 1, 3 channels, 32x32 CIFAR-10 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)