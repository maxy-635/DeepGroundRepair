import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Average pooling layers
        self.avgpool1 = nn.AvgPool2D(kernel_size=1, stride=1, padding=0)
        self.avgpool2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.avgpool3 = nn.AvgPool2D(kernel_size=4, stride=4, padding=0)

        # Flatten layers
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.flatten3 = nn.Flatten()

        # Concatenate layer
        self.concat = nn.Concat()

        # Fully connected layer and reshape
        self.fc1 = nn.Linear(in_features=16*16*3, out_features=1024) # Assuming input is 28x28x1
        self.reshape = nn.Reshape((1, 1024, 4, 4))

        # Second block: Depthwise separable convolutional layers
        self.conv1 = nn.Conv2D(in_channels=1024, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2D(in_channels=1024, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2D(in_channels=1024, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2D(in_channels=1024, out_channels=128, kernel_size=7, stride=1, padding=3)

        # Split layer
        self.split = nn.Split(num_or_sections=4, axis=1)

        # Concatenate layer
        self.concat2 = nn.Concat()

        # Flatten layer
        self.flatten4 = nn.Flatten()

        # Final fully connected layer
        self.fc2 = nn.Linear(in_features=128*4, out_features=10)

    def forward(self, x):
        # First block
        x1 = self.avgpool1(x)
        x2 = self.avgpool2(x)
        x3 = self.avgpool3(x)

        x1 = self.flatten1(x1)
        x2 = self.flatten2(x2)
        x3 = self.flatten3(x3)

        x = self.concat([x1, x2, x3])

        # Fully connected layer and reshape
        x = self.fc1(x)
        x = self.reshape(x)

        # Second block
        x1, x2, x3, x4 = self.split(x)

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)

        x = self.concat2([x1, x2, x3, x4])

        # Flatten and final fully connected layer
        x = self.flatten4(x)
        x = self.fc2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (batch size 1, 28x28 grayscale image)
    input_tensor = paddle.randn(shape=[1, 28, 28, 1])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)