import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super().__init__()

        # First branch
        self.conv1 = nn.Conv2D(channels=3, channels=16, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2D(channels=16, channels=32, kernel_size=1, padding=0)

        # Second branch
        self.conv3 = nn.Conv2D(channels=3, channels=16, kernel_size=1, padding=0)
        self.conv4 = nn.Conv2D(channels=16, channels=32, kernel_size=1, padding=0)
        self.conv5 = nn.Conv2D(channels=32, channels=64, kernel_size=1, padding=0)

        # Concatenation
        self.concat = nn.Concat(dim=1)

        # Final convolution
        self.conv6 = nn.Conv2D(channels=64, channels=3, kernel_size=1, padding=0)

        # Direct connection
        self.direct_conn = nn.Dense(channels=64, channels=64)

        # Fully connected layers
        self.fc1 = nn.Dense(channels=64, channels=128)
        self.fc2 = nn.Dense(channels=128, channels=10)

        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        # First branch
        x1 = self.relu(self.conv1(x))
        x1 = self.relu(self.conv2(x1))

        # Second branch
        x2 = self.relu(self.conv3(x))
        x2 = self.relu(self.conv4(x2))
        x2 = self.relu(self.conv5(x2))

        # Concatenation
        x = self.concat([x1, x2])

        # Final convolution
        x = self.relu(self.conv6(x))

        # Direct connection
        x = x + self.direct_conn(x)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)