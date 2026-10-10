import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block
        self.avg_pool1 = nn.AvgPool2D(kernel_size=1, stride=1, padding=0)
        self.avg_pool2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.avg_pool3 = nn.AvgPool2D(kernel_size=4, stride=4, padding=0)

        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(3 * 64 * 64, 128)  # Adjust the input size based on pooling outputs
        self.reshape1 = nn.ReshapeTransform(in_event_shape=(128, 1, 1, 1))

        # Second block
        self.conv1 = nn.Conv2D(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2D(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2D(in_channels=128, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2D(in_channels=128, out_channels=64, kernel_size=7, stride=1, padding=0)

        self.concat = nn.Concat(dim=1)
        self.flatten2 = nn.Flatten()
        self.fc2 = nn.Linear(64 * 4, 10)  # Adjust the input size based on the second block outputs

    def forward(self, x):
        # First block
        x = self.avg_pool1(x)
        x = self.avg_pool2(x)
        x = self.avg_pool3(x)

        x = self.flatten1(x)
        x = self.fc1(x)
        x = self.reshape1(x)

        # Second block
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.concat(x)
        x = self.flatten2(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 1, 28, 28))  # Example input shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)