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

        # Three parallel blocks
        self.blocks = nn.LayerList([
            nn.Sequential(
                nn.Conv2D(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.BatchNorm2D(num_features=64),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2D(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.BatchNorm2D(num_features=64),
                nn.ReLU()
            ),
            nn.Sequential(
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
        ])

        # Flatten and fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 32 * 32, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)

        # Store initial output for residual connection
        initial_output = x

        # Three parallel blocks
        block_outputs = []
        for block in self.blocks:
            block_output = block(x)
            block_outputs.append(block_output)

        # Sum the outputs of the three blocks
        x = initial_output + sum(block_outputs)

        # Flatten and fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, axis=1)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)