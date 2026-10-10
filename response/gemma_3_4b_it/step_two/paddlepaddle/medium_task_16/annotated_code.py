import paddle
import paddle.nn as nn
import paddle.vision.datasets as datasets
import paddle.io
import paddle.nn.initializer as nn_initializer

class DL_Model(nn.Layer):
    def __init__(self):
        super().__init__()

        # Input channel dimension
        in_channels = 3

        # First Conv layer
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels // 3,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_initializer=nn_initializer.XavierUniform()
        )

        # Average pooling
        self.avg_pool1 = nn.AvgPool2D(
            kernel_size=2,
            stride=2,
            padding=0
        )

        # Second Conv layer
        self.conv2 = nn.Conv2D(
            in_channels=in_channels // 3,
            out_channels=in_channels // 3,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_initializer=nn_initializer.XavierUniform()
        )

        # Average pooling
        self.avg_pool2 = nn.AvgPool2D(
            kernel_size=2,
            stride=2,
            padding=0
        )

        # Third Conv layer
        self.conv3 = nn.Conv2D(
            in_channels=in_channels // 3,
            out_channels=in_channels // 3,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_initializer=nn_initializer.XavierUniform()
        )

        # Average pooling
        self.avg_pool3 = nn.AvgPool2D(
            kernel_size=2,
            stride=2,
            padding=0
        )

        # Concatenate the results
        self.concat = nn.Concat(dim=1)

        # Flatten
        self.flatten = nn.Flatten()

        # First Linear layer
        self.linear1 = nn.Linear(
            in_features=in_channels // 3 * 3,
            out_features=128,
            weight_initializer=nn_initializer.XavierUniform()
        )

        # Second Linear layer
        self.linear2 = nn.Linear(
            in_features=128,
            out_features=10,
            weight_initializer=nn_initializer.XavierUniform()
        )

    def forward(self, x):
        # First Conv layer
        x = self.conv1(x)

        # Average pooling
        x = self.avg_pool1(x)

        # Second Conv layer
        x = self.conv2(x)

        # Average pooling
        x = self.avg_pool2(x)

        # Third Conv layer
        x = self.conv3(x)

        # Average pooling
        x = self.avg_pool3(x)

        # Concatenate
        x = self.concat([x, x, x])

        # Flatten
        x = self.flatten(x)

        # Linear layers
        x = self.linear1(x)
        x = self.linear2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)