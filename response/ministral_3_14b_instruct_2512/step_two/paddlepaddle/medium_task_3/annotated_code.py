import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # First block: Conv2D + MaxPool2D
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Second block: Conv2D + MaxPool2D
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2D(kernel_size='mask', stride='mask', padding='mask')

        # Final Conv2D to match input channel dimension
        self.conv3 = nn.Conv2D(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)

        # Fully connected layer
        self.fc = nn.Linear(in_features=7*7*1, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # First block
        x1 = self.conv1(x)  # output tensor shape: (1, 32, 28, 28)
        x1 = self.pool1(x1)  # output tensor shape: (1, 32, 14, 14)

        # Second block
        x2 = self.conv2(x1)  # output tensor shape: (1, 64, 14, 14)
        x2 = self.pool2(x2)  # output tensor shape of this layer should be (1, 64, 28, 28)

        # Final Conv2D to match input channel dimension
        x3 = self.conv3(x2)

        # Combine with input through addition
        combined = x + x3

        # Flatten and pass through fully connected layer
        combined = combined.flatten(start_axis=1)
        out = self.fc(combined)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 1, 28, 28))  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)