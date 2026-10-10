import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.initial_conv = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Block 1
        self.block1_group1_conv1x1 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.block1_group1_depthwise = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32)
        self.block1_group1_conv1x1_2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Block 2
        self.block2_reshape = None # Placeholder for reshape operation
        self.block2_permute = None # Placeholder for permute operation

        # Flatten and Linear layer
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=32 * 28 * 28, out_features=10)

    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)

        # Block 1
        x1, x2 = paddle.split(x, num_or_sections=2, axis=-1)

        # Group 1
        x1 = self.block1_group1_conv1x1(x1)
        x1 = self.block1_group1_depthwise(x1)
        x1 = self.block1_group1_conv1x1_2(x1)

        # Concatenate
        x = paddle.concat([x1, x2], axis=-1)

        # Block 2
        shape = x.shape
        height, width = shape[2], shape[3]
        groups = 4
        channels_per_group = shape[1] // groups

        x = paddle.reshape_(x, [shape[0], height, width, groups, channels_per_group])
        x = paddle.transpose(x, perm=[0, 2, 3, 4, 1])
        x = paddle.reshape_(x, [shape[0], height, width, shape[1]])

        # Flatten and Linear layer
        x = self.flatten(x)
        x = self.linear(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)