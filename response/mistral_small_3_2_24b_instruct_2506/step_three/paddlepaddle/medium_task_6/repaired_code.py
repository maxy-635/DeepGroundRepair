import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolution
        self.initial_conv = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Parallel blocks
        self.block1 = nn.Sequential(
            nn.Conv2D(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2D(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2D(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64*32*32, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

        # Softmax output
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        # Initial convolution
        out = self.initial_conv(x)

        # Parallel blocks
        block1_out = self.block1(out)
        block2_out = self.block2(out)
        block3_out = self.block3(out)

        # Add outputs
        out = paddle.add(paddle.add(block1_out, block2_out), block3_out)
        out = paddle.add(out, out)  # Add with initial output

        # Flatten and fully connected layers
        out = paddle.flatten(out, start_axis=1)
        out = self.fc1(out)
        out = self.fc2(out)

        # Softmax output
        out = self.softmax(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]