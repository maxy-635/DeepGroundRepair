import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial Convolutional Layer
        self.initial_conv = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.initial_bn = nn.BatchNorm2D(64)
        self.initial_relu = nn.ReLU()

        # Block 1
        self.block1_conv = nn.Conv2D(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.block1_bn = nn.BatchNorm2D(64)
        self.block1_relu = nn.ReLU()

        # Block 2
        self.block2_conv = nn.Conv2D(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.block2_bn = nn.BatchNorm2D(64)
        self.block2_relu = nn.ReLU()

        # Block 3
        self.block3_conv = nn.Conv2D(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.block3_bn = nn.BatchNorm2D(64)
        self.block3_relu = nn.ReLU()

        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(
            in_features=64 * 32 * 32,  # Assuming input size is 32x32
            out_features=128
        )
        self.fc2 = nn.Linear(
            in_features=128,
            out_features=10
        )

        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        # Initial Convolution
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_relu(x)

        # Block 1
        block1_out = self.block1_conv(x)
        block1_out = self.block1_bn(block1_out)
        block1_out = self.block1_relu(block1_out)

        # Block 2
        block2_out = self.block2_conv(x)
        block2_out = self.block2_bn(block2_out)
        block2_out = self.block2_relu(block2_out)

        # Block 3
        block3_out = self.block3_conv(x)
        block3_out = self.block3_bn(block3_out)
        block3_out = self.block3_relu(block3_out)

        # Add outputs of blocks to initial convolution output
        x = x + block1_out + block2_out + block3_out

        # Flatten
        x = self.flatten(x)

        # Fully Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)