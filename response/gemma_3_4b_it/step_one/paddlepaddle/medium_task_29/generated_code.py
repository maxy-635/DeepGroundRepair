import paddle
import paddle.vision.datasets as datasets
import paddle.io
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.metric as metric

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # First max pooling layer
        self.pool1 = nn.MaxPool2D(kernel_size=1, stride=1)
        # Flatten the output of the first convolutional layer and max pooling layer
        self.flatten1 = nn.Flatten()

        # Second convolutional layer
        self.conv2 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Second max pooling layer
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2)
        # Flatten the output of the second convolutional layer and max pooling layer
        self.flatten2 = nn.Flatten()

        # Third convolutional layer
        self.conv3 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Third max pooling layer
        self.pool3 = nn.MaxPool2D(kernel_size=4, stride=4)
        # Flatten the output of the third convolutional layer and max pooling layer
        self.flatten3 = nn.Flatten()

        # Linear layers
        self.fc1 = nn.Linear(in_features=64 * 4 * 4, out_features=128)  # Adjust input size based on pooling
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)

        # Flatten
        x = self.flatten3(x)

        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with the expected shape
    input_tensor = paddle.randn(1, 3, 32, 32)  # Example shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)