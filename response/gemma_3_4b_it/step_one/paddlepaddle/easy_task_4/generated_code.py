import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.functional.max_pool2d(self.conv1(paddle.randn(1, 1, 28, 28)), kernel_size=2, stride=2)

        # Second convolutional block
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.functional.max_pool2d(self.conv2(paddle.randn(1, 32, 14, 14)), kernel_size=2, stride=2)

        # Third convolutional block
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.functional.max_pool2d(self.conv3(paddle.randn(1, 64, 7, 7)), kernel_size=2, stride=2)

        # Fourth convolutional block
        self.conv4 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.functional.max_pool2d(self.conv4(paddle.randn(1, 128, 4, 4)), kernel_size=2, stride=2)

        # Flatten the feature maps
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

        # Softmax layer
        self.softmax = nn.Softmax2D()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.pool4(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        x = self.softmax(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(1, 1, 28, 28)  # Batch size 1, 1 input channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)