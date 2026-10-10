import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial Convolution
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Parallel Blocks
        self.block1 = nn.Layer(
            lambda: nn.Sequential(
                nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2D(32),
                nn.ReLU(),
                nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2D(32),
                nn.ReLU()
            )
        )
        self.block2 = nn.Layer(
            lambda: nn.Sequential(
                nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2D(32),
                nn.ReLU(),
                nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2D(32),
                nn.ReLU()
            )
        )
        self.block3 = nn.Layer(
            lambda: nn.Sequential(
                nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2D(32),
                nn.ReLU(),
                nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2D(32),
                nn.ReLU()
            )
        )

        # Flatten and Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # Adjust the input size based on the expected output size
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Initial Convolution
        x = self.conv1(x)

        # Parallel Blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Add outputs of blocks
        x = x + self.conv1(x)  # Adding the initial convolution's output

        # Flatten
        x = self.flatten(x)

        # Fully Connected Layers
        x = self.fc1(x)
        x = self.fc2(x)

        # Softmax
        x = nn.Softmax(axis=1)(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor
    input_tensor = paddle.randn(batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)