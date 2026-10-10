import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Split input into three groups
        self.split_1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.split_2 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.split_3 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)

        # Concatenate the outputs
        self.concat = nn.Concat(axis='mask')

        # Flatten and fully connected layer
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=32 * 32 * 32, out_features=10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        # Split the input
        x1 = self.split_1(x)
        x2 = self.split_2(x)
        x3 = self.split_3(x)

        # Concatenate the outputs
        x = self.concat([x1, x2, x3])

        # Flatten and pass through the linear layer
        x = self.flatten(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)