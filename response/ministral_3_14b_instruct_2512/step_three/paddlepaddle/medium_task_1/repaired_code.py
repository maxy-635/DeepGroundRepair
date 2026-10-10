import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        # Second convolutional layer
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        # Flatten layer
        self.flatten = nn.Flatten()
        # First fully connected layer
        self.fc1 = nn.Linear(in_features=64*8*8, out_features=512)
        # Second fully connected layer
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First convolutional layer
        x1 = self.conv1(x)  # output tensor shape: (1, 32, 32, 32)
        # Second convolutional layer
        x2 = self.conv2(x1)  # output tensor shape of this layer should be (1, 3, 32, 32)
        # Max pooling layer
        x3 = self.pool(x2)  # output tensor shape of this layer should be (1, 64, 32, 32)
        # Add input to output features
        x4 = x + x3
        # Flatten the features
        x5 = self.flatten(x4)
        # First fully connected layer
        x6 = self.fc1(x5)
        # Second fully connected layer
        x7 = self.fc2(x6)
        return x7

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)