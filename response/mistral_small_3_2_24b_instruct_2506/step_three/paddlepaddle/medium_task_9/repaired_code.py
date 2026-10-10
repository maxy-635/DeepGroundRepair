import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolutional layer
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

        # First basic block
        self.basic_block1 = nn.Sequential(
            nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=16),
            nn.ReLU()
        )

        # Second basic block
        self.basic_block2 = nn.Sequential(
            nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=16),
            nn.ReLU()
        )

        # Branch convolutional layer
        self.branch_conv = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Average pooling layer
        self.avg_pool = nn.AvgPool2D(kernel_size=8, stride=1, padding=0)

        # Fully connected layer
        self.fc = nn.Linear(in_features=10000, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Initial convolution
        x = self.conv1(x)  # output tensor shape: (1, 16, 32, 32)

        # First basic block
        residual = x  # output tensor shape: (1, 16, 32, 32)
        x = self.basic_block1(x)  # output tensor shape: (1, 16, 32, 32)
        x = paddle.add(x, residual)  # output tensor shape: (1, 16, 32, 32)

        # Second basic block
        residual = x  # output tensor shape: (1, 16, 32, 32)
        x = self.basic_block2(x)  # output tensor shape: (1, 16, 32, 32)
        x = paddle.add(x, residual)  # output tensor shape: (1, 16, 32, 32)

        # Branch
        branch = self.branch_conv(x)  # output tensor shape: (1, 16, 32, 32)
        x = paddle.add(x, branch)  # output tensor shape: (1, 16, 32, 32)

        # Average pooling and flatten
        x = self.avg_pool(x)  # output tensor shape: (1, 16, 25, 25)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)  # output tensor shape: (1, 10000)

        # Fully connected layer
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # CIFAR-10 image shape: [batch_size, 3, 32, 32]
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]