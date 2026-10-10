import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: two convolution layers followed by max pooling
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Second block: two convolution layers followed by max pooling
        self.conv3 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Third block: three convolution layers followed by max pooling
        self.conv5 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Fourth block: three convolution layers followed by max pooling
        self.conv8 = nn.Conv2D(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2D(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # First block
        x = self.conv1(x)  # output tensor shape: (1, 32, 28, 28)
        x = self.conv2(x)  # output tensor shape: (1, 32, 28, 28)
        x = self.pool1(x)  # output tensor shape: (1, 32, 14, 14)

        # Second block
        x = self.conv3(x)  # output tensor shape: (1, 64, 14, 14)
        x = self.conv4(x)  # output tensor shape: (1, 64, 14, 14)
        x = self.pool2(x)  # output tensor shape: (1, 64, 7, 7)

        # Third block
        x = self.conv5(x)  # output tensor shape: (1, 128, 7, 7)
        x = self.conv6(x)  # output tensor shape: (1, 128, 7, 7)
        x = self.conv7(x)  # output tensor shape: (1, 128, 7, 7)
        x = self.pool3(x)  # output tensor shape: (1, 128, 3, 3)

        # Fourth block
        x = self.conv8(x)  # output tensor shape: (1, 256, 3, 3)
        x = self.conv9(x)  # output tensor shape: (1, 256, 3, 3)
        x = self.conv10(x)  # output tensor shape: (1, 256, 3, 3)
        x = self.pool4(x)  # output tensor shape: (1, 256, 1, 1)

        # Flatten and fully connected layers
        x = self.flatten(x)  # output tensor shape: (1, 256)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 1, 28, 28))  # Batch size of 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]