import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Split into 3 groups with 1x1 conv and ReLU
        self.block1_conv1 = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.block1_conv2 = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.block1_conv3 = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

        # Block 2: Channel shuffling
        # We'll use LayerNorm for normalization before reshaping
        self.ln = nn.LayerNorm([32, 32, 3])  # Assuming input size is 32x32 for CIFAR-10

        # Block 3: 3x3 depthwise separable convolution
        self.depthwise_conv = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, groups=3)
        self.pointwise_conv = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Branch path: average pooling
        self.avg_pool = nn.functional.avg_pool2d

        # Final fully connected layer
        self.fc = nn.Linear(in_features=3*32*32, out_features=10)  # Adjust in_features based on actual output size

    def forward(self, x):
        # Block 1 processing
        x1 = self.block1_conv1(x)
        x2 = self.block1_conv2(x)
        x3 = self.block1_conv3(x)

        # Apply ReLU to each branch
        x1 = self.relu(x1)
        x2 = self.relu(x2)
        x3 = self.relu(x3)

        # Concatenate the three branches
        block1_out = paddle.concat([x1, x2, x3], axis=1)

        # Block 2: Channel shuffling
        # Normalize first
        block1_out = self.ln(block1_out)

        # Get shape information
        batch_size, channels, height, width = block1_out.shape
        groups = 3
        channels_per_group = channels // groups

        # Reshape to (batch_size, height, width, groups, channels_per_group)
        reshaped = paddle.reshape(block1_out, [batch_size, height, width, groups, channels_per_group])

        # Permute dimensions to swap groups and channels_per_group
        permuted = paddle.transpose(reshaped, [0, 1, 2, 4, 3])

        # Reshape back to original shape
        shuffled = paddle.reshape(permuted, [batch_size, channels, height, width])

        # Block 3: Depthwise separable convolution
        depthwise_out = self.depthwise_conv(shuffled)
        pointwise_out = self.pointwise_conv(depthwise_out)

        # Branch path: average pooling
        branch_out = self.avg_pool(x, kernel_size=2, stride=2, padding=0)

        # Concatenate main path and branch path
        combined = paddle.concat([pointwise_out, branch_out], axis=1)

        # Flatten and pass through fully connected layer
        flattened = paddle.flatten(combined, start_axis=1)
        output = self.fc(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, channels, height, width]
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)