import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.initial_conv = nn.Conv2D(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Block 1 components
        self.conv1x1_1 = nn.Conv2D(
            in_channels=32,
            out_channels=16,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.depthwise_conv = nn.Conv2D(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=16  # Depthwise convolution
        )
        self.conv1x1_2 = nn.Conv2D(
            in_channels=16,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Block 2 components
        self.channels_per_group = 8  # Since we're splitting into 4 groups

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=32*28*28, out_features=10)  # MNIST has 28x28 images

    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)
        x = F.prelu(x)

        # Block 1
        # Split into two groups along channel dimension
        x1, x2 = paddle.split(x, num_or_sections=2, axis=1)

        # Process first group
        x1 = self.conv1x1_1(x1)
        x1 = F.prelu(x1)
        x1 = self.depthwise_conv(x1)
        x1 = F.prelu(x1)
        x1 = self.conv1x1_2(x1)
        x1 = F.prelu(x1)

        # Concatenate with second group (unchanged)
        x = paddle.concat([x1, x2], axis=1)

        # Block 2 - Channel shuffling
        batch_size, channels, height, width = x.shape
        groups = 4
        channels_per_group = channels // groups

        # Reshape to (batch, height, width, groups, channels_per_group)
        x = paddle.reshape(x, [batch_size, height, width, groups, channels_per_group])

        # Permute dimensions to swap groups and channels
        x = paddle.transpose(x, perm=[0, 1, 2, 4, 3])

        # Reshape back to original shape
        x = paddle.reshape(x, [batch_size, channels, height, width])

        # Final classification
        x = self.flatten(x)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, 1, 28, 28] for MNIST
    input_tensor = paddle.randn([32, 1, 28, 28])  # batch_size=32
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # Should be [32, 10]