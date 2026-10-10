import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # 1x1 convolution layer to reduce input dimensionality
        self.conv1 = nn.Conv2D(in_channels=1,
                                out_channels=16,
                                kernel_size=1,
                                stride=1,
                                padding=0)
        # 3x3 convolutional layer to extract features
        self.conv2 = nn.Conv2D(in_channels=16,
                                out_channels=32,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        # 1x1 convolution layer to restore dimensionality
        self.conv3 = nn.Conv2D(in_channels=32,
                                out_channels=16,
                                kernel_size=1,
                                stride=1,
                                padding=0)
        # Flatten layer
        self.flatten = nn.Flatten()
        # Fully connected layer for classification
        self.linear = nn.Linear(in_features=16 * 7 * 7,
                                 out_features=10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass through convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        # Flatten the output
        x = self.flatten(x)
        # Pass through the fully connected layer
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor with shape (1, 1, 28, 28)
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)