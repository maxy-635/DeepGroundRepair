import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial convolution layer
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Define the three parallel blocks
        self.block1 = nn.Sequential(
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(num_channels=64),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(num_channels=64),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm(num_channels=64),
            nn.ReLU()
        )

        # Flatten and fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=65536, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Initial convolution
        x = self.conv1(x)  # output tensor shape: (1, 64, 32, 32)

        # Add outputs from parallel blocks to initial convolution
        x = paddle.add(x, self.block1(x))  # output tensor shape: (1, 64, 32, 32)
        x = paddle.add(x, self.block2(x))  # output tensor shape: (1, 64, 32, 32)
        x = paddle.add(x, self.block3(x))  # output tensor shape: (1, 64, 32, 32)

        # Flatten and pass through fully connected layers
        x = self.flatten(x)  # output tensor shape: (1, 65536)
        x = paddle.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return paddle.nn.functional.softmax(x, axis=1)

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Example input shape for CIFAR-10
    output_tensor = model(input_tensor)
    print(output.out_shape)  # Note: There was a typo in the original code
    print(output_tensor.shape)