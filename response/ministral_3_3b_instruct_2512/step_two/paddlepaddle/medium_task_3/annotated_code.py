import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First convolutional block
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Second convolutional block
        self.conv2 = nn.Conv2D(in_channels=32, out_channels='mask', kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2D(kernel_size='mask', stride='mask', padding='mask')

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layer
        self.fc = nn.Linear(in_features=32 * 14 * 14, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # First convolutional block
        x1 = self.conv1(x)  # output tensor shape: (1, 32, 28, 28)
        x1 = paddle.nn.ReLU()(x1)  # output tensor shape: (1, 32, 28, 28)
        x1 = self.pool1(x1)  # output tensor shape: (1, 32, 14, 14)

        # Second convolutional block
        x2 = self.conv2(x1)  # output tensor shape of this layer should be (1, 1, 14, 14)
        x2 = paddle.nn.ReLU()(x2)
        x2 = self.pool2(x2)  # output tensor shape of this layer should be (1, 32, 14, 14)

        # Add the original input to the output of the second convolutional block
        x = paddle.add(x, x2)

        # Flatten and pass through fully connected layer
        x = self.flatten(x)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    input_tensor = paddle.zeros(shape=(1, 1, 28, 28))
    model = DL_Model()
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)
    print("Output values:", output_tensor.numpy())