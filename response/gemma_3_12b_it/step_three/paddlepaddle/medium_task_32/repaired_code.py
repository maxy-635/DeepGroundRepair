import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1x1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.celu1 = nn.CELU(alpha=1.0)
        self.celu2 = nn.CELU(alpha=1.0)
        self.celu3 = nn.CELU(alpha=1.0)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=98304, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x1 = self.conv1x1(x)  # output tensor shape: (1, 32, 32, 32)
        x1 = self.celu1(x1)  # output tensor shape: (1, 32, 32, 32)
        x2 = self.conv3x3(x)  # output tensor shape: (1, 32, 32, 32)
        x2 = self.celu2(x2)  # output tensor shape: (1, 32, 32, 32)
        x3 = self.conv5x5(x)  # output tensor shape: (1, 32, 32, 32)
        x3 = self.celu3(x3)  # output tensor shape: (1, 32, 32, 32)
        fused = paddle.concat([x1, x2, x3], axis=1)  # output tensor shape: (1, 96, 32, 32)
        flattened = self.flatten(fused)  # output tensor shape: (1, 98304)
        output = self.fc(flattened)
        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)