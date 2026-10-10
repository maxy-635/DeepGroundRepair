import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Block 1: Splitting and 1x1 convolutions
        self.conv1_1 = nn.Conv2D(3, 1, kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2D(3, 1, kernel_size=1, stride=1, padding=0)
        self.conv1_3 = nn.Conv2D(3, 1, kernel_size=1, stride=1, padding=0)

        # Concatenation layer
        self.concat = nn.LayerNorm([3])

        # Block 2: Channel Shuffling
        self.permute = nn.LayerNorm([3])
        self.flatten = nn.Flatten()
        self.reshape = paddle.distribution.ReshapeTransform([None, 3, 3, 10])

        # Depthwise separable convolution
        self.depthwise_conv = nn.Conv2D(3, 3, kernel_size=3, stride=1, padding=1, groups=3)

        # Branch for direct connection
        self.branch_conv = nn.Conv2D(3, 3, kernel_size=1, stride=1, padding=0)

        # Fully connected layer
        self.fc = nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        # Block 1: Splitting and 1x1 convolutions
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x3 = self.conv1_3(x)

        # Concatenation
        x_concat = paddle.concat([x1, x2, x3], axis=1)
        x_concat = self.concat(x_concat)

        # Block 2: Channel Shuffling
        # Reshape to (height, width, groups, channels_per_group)
        x_reshaped = self.reshape(x_concat)
        x_permuted = paddle.permute(x_reshaped, [0, 1, 3, 2])
        x_shuffled = paddle.reshape(x_permuted, [x_concat.shape[0], 32, 32, 3, 3])

        # Depthwise separable convolution
        x_depthwise = self.depthwise_conv(x_shuffled)

        # Branch for direct connection
        x_branch = self.branch_conv(x)

        # Combine outputs
        x_combined = x_depthwise + x_branch

        # Flatten and fully connected layer
        x_flattened = self.flatten(x_combined)
        x_flattened = x_flattened.reshape([-1, 3 * 32 * 32])
        output = self.fc(x_flattened)

        return output

if __name__ == "__main__":
    # Dummy input tensor for CIFAR-10 (batch_size=1, channels=3, height=32, width=32)
    input_tensor = paddle.randn([1, 3, 32, 32])
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)