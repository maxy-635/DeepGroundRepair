import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolutional layer to reduce dimensionality to 16
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

        # First basic block
        self.basic_block1 = nn.Sequential(
            nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(num_channels=16),
            nn.ReLU(),
            nn.Identity()  # Branch path
        )

        # Second basic block with branch
        self.basic_block2 = nn.Sequential(
            nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(num_channels=16),
            nn.ReLU(),
            nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)  # Branch path
        )

        # Average pooling layer
        self.avg_pool = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)

        # Fully connected layer
        self.fc = nn.Linear(in_features=16 * 8 * 8, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Initial convolutional layer
        x = self.conv1(x)  # output tensor shape: (1, 16, 32, 32)

        # First basic block
        x_main = self.basic_block1(x)  # output tensor shape: (1, 16, 32, 32)
        x_branch = x  # output tensor shape: (1, 16, 32, 32)
        x = paddle.add(x_main, x_branch)  # output tensor shape: (1, 16, 32, 32)

        # Second basic block with branch
        x_main = self.basic_block2(x)  # output tensor shape: (1, 16, 32, 32)
        x_branch = self.basic_block2.conv2(x, padding=1)
        x = paddle.add(x_main, x_branch)

        # Average pooling
        x = self.avg_pool(x)
        x = x.flatten(start_dim=1)

        # Fully connected layer
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Example input shape for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)