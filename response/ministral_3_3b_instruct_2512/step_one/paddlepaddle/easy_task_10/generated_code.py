import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=2)
        self.depthwise_conv = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=2, groups=32)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=10, kernel_size=1, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=10, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = paddle.nn.ReLU()(x)
        x = self.depthwise_conv(x)
        x = paddle.nn.ReLU()(x)
        x = self.conv2(x)
        x = paddle.nn.ReLU()(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)