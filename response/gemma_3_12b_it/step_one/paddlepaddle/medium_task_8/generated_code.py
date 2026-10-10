import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.conv1_main = nn.Conv2D(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv2_main = nn.Conv2D(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Branch path
        self.conv1_branch = nn.Conv2D(
            in_channels=3,
            out_channels=16,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Flatten and Linear
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(
            in_features=64 * 32 * 32,  # Assuming input shape is (3, 32, 32)
            out_features=10
        )

    def forward(self, x):
        # Main path
        x1, x2, x3 = paddle.split(x, 3, axis=-1)  # Split along the last dimension
        x2 = self.conv1_main(x2)
        x3 = F.relu(x3)
        x_combined = paddle.concat([x2, x3], axis=-1)
        x_main = self.conv2_main(x_combined)
        x_main = paddle.concat([x1, x_main], axis=-1)

        # Branch path
        x_branch = self.conv1_branch(x)

        # Fusion
        combined_output = paddle.add(x_main, x_branch)

        # Flatten and Linear
        flattened_output = self.flatten(combined_output)
        output = self.linear(flattened_output)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)