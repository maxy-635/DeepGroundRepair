import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Depthwise separable convolution
        self.depthwise_conv = nn.Conv2D(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.pointwise_conv = nn.Conv2D(
            in_channels=32,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )
        # Dropout layers
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.25)
        # Flatten and fully connected layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(
            in_features=64 * 28 * 28,
            out_features=10
        )

    def forward(self, x):
        # Depthwise separable convolution
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = paddle.nn.functional.relu(x)
        x = self.dropout1(x)
        # Second depthwise separable convolution
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = paddle.nn.functional.relu(x)
        x = self.dropout2(x)
        # Flatten and fully connected
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)