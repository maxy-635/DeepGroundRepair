import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)

        # Second convolutional block
        self.conv2 = nn.Conv2D(in_channels=32, out_channels='mask', kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2D(kernel_size='mask', stride='mask', padding='mask')

        # Fully connected layer
        self.linear = nn.Linear(in_features=32 * 7 * 7, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Pass through the first convolutional block
        x1 = self.conv1(x)  # output tensor shape: (1, 32, 28, 28)
        x1 = self.pool1(x1)  # output tensor shape: (1, 32, 14, 14)

        # Pass through the second convolutional block
        x2 = self.conv2(x1)  # output tensor shape of this layer should be (1, 1, 14, 14)
        x2 = self.pool2(x2)  # output tensor shape of this layer should be (1, 32, 14, 14)

        # Combine with the input
        x = x + x2

        # Flatten the tensor for the fully connected layer
        x = paddle.reshape(x, [x.shape[0], 32 * 7 * 7])

        # Pass through the fully connected layer
        x = self.linear(x)

        # Apply softmax to get probability distribution
        x = F.softmax(x, axis=1)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])  # Batch size of 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)