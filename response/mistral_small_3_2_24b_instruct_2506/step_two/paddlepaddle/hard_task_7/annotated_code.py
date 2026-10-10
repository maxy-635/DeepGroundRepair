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
        self.conv1x1_1 = nn.Conv2D(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')






        self.depthwise_conv = nn.Conv2D(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=32
        )
        self.conv1x1_2 = nn.Conv2D(
            in_channels=32,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Block 2 components
        self.groups = 4

        # Fully connected layer
        self.fc = nn.Linear(
            in_features=32 * 7 * 7,
            out_features=10
        )

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Initial convolution
        x = self.initial_conv(x)  # output tensor shape: (1, 32, 28, 28)

        # Block 1
        # Split the input into two groups
        x1, x2 = paddle.split(x, num_or_sections=2, axis=1)  # output tensor shape: (1, 16, 28, 28); (1, 16, 28, 28)

        # Process the first group
        x1 = self.conv1x1_1(x1)
        x1 = self.depthwise_conv(x1)
        x1 = self.conv1x1_2(x1)

        # Merge the two groups
        x = paddle.concat([x1, x2], axis=1)

        # Block 2 - Channel shuffling
        # Get the shape of the input
        shape = paddle.shape(x)
        height, width = shape[2], shape[3]
        channels = shape[1]

        # Reshape into groups
        x = paddle.reshape(x, shape=[-1, height, width, self.groups, channels // self.groups])

        # Permute dimensions
        x = paddle.transpose(x, perm=[0, 1, 2, 4, 3])

        # Reshape back to original shape
        x = paddle.reshape(x, shape=[-1, channels, height, width])

        # Flatten and pass through fully connected layer
        x = paddle.flatten(x, start_axis=1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])  # MNIST image shape: (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]