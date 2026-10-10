import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        # First max pooling layer
        self.maxpool1 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        # Second convolutional layer
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # Second max pooling layer
        self.maxpool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        # Flatten layer
        self.flatten = nn.Flatten()
        # First fully connected layer
        self.linear1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        # Second fully connected layer
        self.linear2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for MNIST (1, 28, 28)
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)