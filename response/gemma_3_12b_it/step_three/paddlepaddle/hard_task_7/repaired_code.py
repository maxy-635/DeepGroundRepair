import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.conv_initial = nn.Conv2D(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Block 1
        self.conv1_1 = nn.Conv2D(
            in_channels=32,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.depthwise_conv = nn.Conv2D(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=32
        )
        self.conv1_2 = nn.Conv2D(
            in_channels=32,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Block 2
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(
            in_features=32 * 7 * 7,
            out_features=10
        )

    def forward(self, x):
        # Initial convolutional layer
        x = self.conv_initial(x)

        # Block 1
        x1, x2 = paddle.split(x, [x.shape[-1] // 2, x.shape[-1] // 2], axis=-1)
        x1 = self.conv1_1(x1)
        x1 = self.depthwise_conv(x1)
        x1 = self.conv1_2(x1)
        x = paddle.concat([x1, x2], axis=-1)

        # Block 2
        shape = paddle.shape(x)
        height, width, channels = shape[1], shape[2], shape[3]
        groups = 4
        channels_per_group = channels // groups

        x = paddle.reshape(x, [shape[0], height, width, groups, channels_per_group])
        x = paddle.transpose(x, [0, 1, 2, 4, 3])
        x = paddle.reshape(x, [shape[0], height, width, channels])

        # Flatten and fully connected layer
        x = self.flatten(x)
        x = self.linear(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)