import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # 1x1 convolutional layer to increase dimensionality
        self.conv1x1_expand = nn.Conv2D(
            in_channels=1,
            out_channels=32,
            kernel_size=1,
            stride=2,
            padding=0
        )
        # 3x3 depthwise separable convolutional layer for feature extraction
        self.depthwise_conv = nn.Conv2D(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=32
        )
        # 1x1 convolutional layer to reduce dimensionality
        self.conv1x1_reduce = nn.Conv2D(
            in_channels=32,
            out_channels=10,
            kernel_size=1,
            stride=2,
            padding=0
        )
        # Flattening layer
        self.flatten = nn.Flatten()
        # Fully connected layer for classification
        self.fc = nn.Linear(in_features='mask', out_features=10)




    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        x = self.conv1x1_expand(x)  # output tensor shape: (1, 32, 14, 14)
        x = self.depthwise_conv(x)  # output tensor shape: (1, 32, 7, 7)
        x = self.conv1x1_reduce(x)  # output tensor shape: (1, 10, 4, 4)
        x = self.flatten(x)  # output tensor shape: (1, 160)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 1, 28, 28))  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]