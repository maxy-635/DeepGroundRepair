import paddle
import paddle.vision.datasets as datasets
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Convolutional Layer 1
        self.conv1 = nn.Conv2D(
            in_channels=3,  # CIFAR-10 has 3 channels (RGB)
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Max-Pooling Layer 1
        self.pool1 = nn.MaxPool2D(
            kernel_size=2,
            stride=2,
            padding=0
        )

        # Convolutional Layer 2
        self.conv2 = nn.Conv2D(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Max-Pooling Layer 2
        self.pool2 = nn.MaxPool2D(
            kernel_size=2,
            stride=2,
            padding=0
        )

        # Add input and output of conv2
        self.add = nn.Add()

        # Flatten
        self.flatten = nn.Flatten()

        # Fully Connected Layer 1
        self.fc1 = nn.Linear(
            in_features=32 * 8 * 8,  # Calculate the size of the flattened tensor
            out_features=64
        )

        # Fully Connected Layer 2
        self.fc2 = nn.Linear(
            in_features=64,
            out_features=10
        )

        # Softmax
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        # Convolutional Layer 1
        x = self.conv1(x)
        x = paddle.nn.relu(x)
        x = self.pool1(x)

        # Convolutional Layer 2
        x = self.conv2(x)
        x = paddle.nn.relu(x)
        x = self.pool2(x)

        # Add input and output of conv2
        x = self.add(x, x)

        # Flatten
        x = self.flatten(x)

        # Fully Connected Layer 1
        x = self.fc1(x)
        x = paddle.nn.relu(x)

        # Fully Connected Layer 2
        x = self.fc2(x)

        # Softmax
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    input_tensor = paddle.randn(1, 3, 32, 32, dtype=paddle.float32)  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: (1, 10)