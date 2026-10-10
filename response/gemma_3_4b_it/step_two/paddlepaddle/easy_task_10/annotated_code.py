import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features='mask', out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        x = self.conv1(x)  # output tensor shape: (1, 16, 13, 13)
        x = self.conv2(x)  # output tensor shape: (1, 32, 6, 6)
        x = self.flatten(x)  # output tensor shape: (1, 1152)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)