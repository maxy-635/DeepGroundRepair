import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1x1 = nn.Conv2D(3, 64, kernel_size=1, stride=1, padding=0)
        self.conv3x3_1 = nn.Conv2D(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3x3_2 = nn.Conv2D(in_channels=64, out_channels='mask', kernel_size=3, stride=1, padding=1)
        self.layer_norm = nn.LayerNorm(normalized_shape=64)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x1 = self.conv1x1(x)  # output tensor shape: (1, 64, 32, 32)
        x2 = self.conv1x1(x)  # output tensor shape: (1, 64, 32, 32)
        x2 = self.conv3x3_1(x2)  # output tensor shape: (1, 64, 32, 32)
        x3 = self.conv1x1(x)  # output tensor shape: (1, 64, 32, 32)
        x3 = self.conv3x3_1(x3)  # output tensor shape: (1, 64, 32, 32)
        x3 = self.conv3x3_2(x3)  # output tensor shape of this layer should be (1, 64, 32, 32)
        x = x + x1 + x2 + x3
        x = self.layer_norm(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)