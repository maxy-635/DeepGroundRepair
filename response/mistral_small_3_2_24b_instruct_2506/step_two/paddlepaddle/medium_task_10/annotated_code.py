import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial convolution to adjust input feature dimensionality to 16
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

        # First level: Basic block
        self.basic_block1 = nn.Sequential(
            nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=16),
            nn.ReLU()
        )

        # Second level: Two residual blocks
        self.residual_block1 = nn.Sequential(
            nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2D(num_features=32),
            nn.ReLU()
        )
        self.residual_block2 = nn.Sequential(
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=32),
            nn.ReLU()
        )
        self.branch_conv1 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=1, stride=2, padding=0)

        self.residual_block3 = nn.Sequential(
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )
        self.residual_block4 = nn.Sequential(
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )
        self.branch_conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=1, stride=2, padding=0)

        # Third level: Global branch
        self.global_branch = nn.Conv2D(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')

        # Final layers
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.fc = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Initial convolution
        x = self.conv1(x)  # output tensor shape: (1, 16, 32, 32)

        # First level: Basic block
        residual = x  # output tensor shape: (1, 16, 32, 32)
        x = self.basic_block1(x)  # output tensor shape: (1, 16, 32, 32)
        x = paddle.add(x, residual)  # output tensor shape: (1, 16, 32, 32)

        # Second level: First residual block
        residual = self.branch_conv1(residual)  # output tensor shape: (1, 32, 16, 16)
        x = self.residual_block1(x)  # output tensor shape: (1, 32, 16, 16)
        x = paddle.add(x, residual)  # output tensor shape: (1, 32, 16, 16)

        # Second level: Second residual block
        residual = x  # output tensor shape: (1, 32, 16, 16)
        x = self.residual_block2(x)  # output tensor shape: (1, 32, 16, 16)
        x = paddle.add(x, residual)  # output tensor shape: (1, 32, 16, 16)

        # Third level: First residual block
        residual = self.branch_conv2(x)  # output tensor shape: (1, 64, 8, 8)
        x = self.residual_block3(x)  # output tensor shape: (1, 64, 8, 8)
        x = paddle.add(x, residual)  # output tensor shape: (1, 64, 8, 8)

        # Third level: Second residual block
        residual = x  # output tensor shape: (1, 64, 8, 8)
        x = self.residual_block4(x)  # output tensor shape: (1, 64, 8, 8)
        x = paddle.add(x, residual)  # output tensor shape: (1, 64, 8, 8)

        # Global branch
        global_branch = self.global_branch(x)
        x = paddle.add(x, global_branch)

        # Final layers
        x = self.avg_pool(x)
        x = paddle.flatten(x, start_axis=1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)