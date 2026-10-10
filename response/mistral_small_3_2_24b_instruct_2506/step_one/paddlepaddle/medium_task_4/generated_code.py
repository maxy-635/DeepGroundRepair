import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path1: Two blocks of convolution followed by average pooling
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)

        # Path2: Single convolutional layer
        self.conv3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Flatten and fully connected layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=64*8*8 + 64*32*32, out_features=10)

    def forward(self, x):
        # Path1
        x1 = self.conv1(x)
        x1 = self.pool1(x1)
        x1 = self.conv2(x1)
        x1 = self.pool2(x1)

        # Path2
        x2 = self.conv3(x)

        # Combine features
        x = paddle.concat([x1, x2], axis=1)

        # Flatten and classify
        x = self.flatten(x)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # CIFAR-10 image shape: [batch_size, 3, 32, 32]
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]