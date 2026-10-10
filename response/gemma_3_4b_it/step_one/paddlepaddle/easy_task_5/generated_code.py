import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Convolutional layer 1 (1x1)
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2D(num_features=16)
        self.relu1 = nn.ReLU()

        # Convolutional layer 2 (3x3)
        self.conv2 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2D(num_features=32)
        self.relu2 = nn.ReLU()

        # Convolutional layer 3 (1x1)
        self.conv3 = nn.Conv2D(in_channels=32, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2D(num_features=16)
        self.relu3 = nn.ReLU()

        # Flatten
        self.flatten = nn.Flatten()

        # Linear layer
        self.linear = nn.Linear(in_features=16 * 7 * 7, out_features=10)  # Adjust input size based on conv layers

        # Softmax
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        # Convolutional layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Convolutional layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        # Convolutional layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # Flatten
        x = self.flatten(x)

        # Linear layer
        x = self.linear(x)

        # Softmax
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 1, 28, 28))  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)