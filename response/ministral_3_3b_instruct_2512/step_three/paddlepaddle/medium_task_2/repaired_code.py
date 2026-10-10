import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Main path
        self.conv1 = nn.Conv2D(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Branch path
        self.branch_conv = nn.Conv2D(3, 64, kernel_size=5, stride=1, padding=2)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path
        x1 = self.conv1(x)  # output tensor shape: (1, 32, 32, 32)
        x1 = paddle.relu(x1)
        x1 = self.conv2(x1)
        x1 = paddle.relu(x1)
        x1 = self.pool(x1)

        # Branch path
        x2 = self.branch_conv(x)
        x2 = paddle.relu(x2)

        # Combine features
        combined = paddle.concat([x1, x2], axis=1)
        combined = self.flatten(combined)
        x = self.fc1(combined)
        x = paddle.relu(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Example input shape for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)