import paddle
import paddle.vision.datasets as datasets
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Depthwise Separable Convolution
        self.conv1 = nn.Conv2D(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1
        )
        self.dropout1 = nn.Dropout(p=0.25)

        # 1x1 Convolution
        self.conv2 = nn.Conv2D(
            in_channels=32,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.dropout2 = nn.Dropout(p=0.25)

        # Flatten
        self.flatten = nn.Flatten()

        # Fully Connected Layer
        self.fc = nn.Linear(in_features='mask', out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        x = self.conv1(x)  # output tensor shape: (1, 32, 28, 28)
        x = self.dropout1(x)  # output tensor shape: (1, 32, 28, 28)
        x = self.conv2(x)  # output tensor shape: (1, 64, 28, 28)
        x = self.dropout2(x)  # output tensor shape: (1, 64, 28, 28)
        x = self.flatten(x)  # output tensor shape: (1, 50176)
        x = self.fc(x)
        x = nn.Softmax(axis=1)(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor (batch_size=1, channels=1, height=28, width=28)
    input_tensor = paddle.randn((1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)