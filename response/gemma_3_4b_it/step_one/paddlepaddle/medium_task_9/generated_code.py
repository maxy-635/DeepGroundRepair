import paddle
import paddle.nn as nn

class DL_Model(paddle.nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Basic block 1
        self.basic_block1 = nn.Layer(
            lambda x: nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)(x) +
                      nn.BatchNorm2D(16) +
                      nn.ReLU()(x)
        )

        # Basic block 2
        self.basic_block2 = nn.Layer(
            lambda x: nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)(x) +
                      nn.BatchNorm2D(16) +
                      nn.ReLU()(x)
        )

        # Branch convolutional layer
        self.branch_conv = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Average pooling layer
        self.avg_pool = nn.AvgPool2D(kernel_size=4, stride=1, padding=2)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layer
        self.fc = nn.Linear(16 * 4 * 4, 10)  # Assuming 10 classes for CIFAR-10

    def forward(self, x):
        # Initial convolutional layer
        x = self.conv1(x)

        # Main path
        x = self.basic_block1(x)
        x = self.basic_block2(x)

        # Branch path
        branch = self.branch_conv(x)

        # Feature fusion
        x = x + branch
        x = self.add(x, branch)

        # Average pooling
        x = self.avg_pool(x)

        # Flatten
        x = self.flatten(x)

        # Fully connected layer
        x = self.fc(x)

        return x

    def add(self, x, y):
        return paddle.add(x, y)


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for validation
    input_tensor = paddle.randn(batch_size=1, in_channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)