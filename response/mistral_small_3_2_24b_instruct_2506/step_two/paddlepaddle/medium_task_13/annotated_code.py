import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size='mask', stride='mask', padding='mask')






        # Second convolutional layer
        self.conv2 = nn.Conv2D(
            in_channels=64,  # 32 (from conv1) + 3 (input channels)
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        # Third convolutional layer
        self.conv3 = nn.Conv2D(
            in_channels=128,  # 64 (from conv2) + 3 (input channels)
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First convolutional layer with concatenation
        x1 = self.conv1(x)  # output tensor shape of this layer should be (1, 32, 32, 32)
        x1 = F.relu(x1)
        x1 = paddle.nn.functional.max_pool2d(x1, kernel_size=2, stride=2)
        x1 = paddle.concat([x, x1], axis=1)

        # Second convolutional layer with concatenation
        x2 = self.conv2(x1)
        x2 = F.relu(x2)
        x2 = paddle.nn.functional.max_pool2d(x2, kernel_size=2, stride=2)
        x2 = paddle.concat([x1, x2], axis=1)

        # Third convolutional layer with concatenation
        x3 = self.conv3(x2)
        x3 = F.relu(x3)
        x3 = paddle.nn.functional.max_pool2d(x3, kernel_size=2, stride=2)

        # Flatten the output
        x3 = paddle.flatten(x3, start_axis=1)

        # Fully connected layers
        x3 = self.fc1(x3)
        x3 = F.relu(x3)
        x3 = self.fc2(x3)

        return x3

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # CIFAR-10 image shape: [3, 32, 32]
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]