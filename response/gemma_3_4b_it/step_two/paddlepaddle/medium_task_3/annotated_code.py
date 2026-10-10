import paddle
import paddle.vision.datasets as datasets
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        self.conv1 = nn.Conv2D(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2D(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=1)
        self.relu = nn.CELU(alpha='mask')
        self.fc = nn.Linear(32 * 7 * 7, 10)  # Adjust input size based on pooling
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        x = self.relu(self.layer_norm1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.reshape(x.shape[0], -1)
        x = paddle.add(x, x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 1, 28, 28))  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor)