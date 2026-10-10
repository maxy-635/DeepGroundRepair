import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.pool1 = nn.AvgPool2D(kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.pool2 = nn.AvgPool2D(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.pool3 = nn.AvgPool2D(kernel_size=(4, 4), stride=(4, 4), padding=(0, 0))
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten()
        self.linear1 = nn.Linear(in_features=4032, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x1 = self.pool1(x)  # output tensor shape: (1, 3, 32, 32)
        x2 = self.pool2(x)  # output tensor shape: (1, 3, 16, 16)
        x3 = self.pool3(x)  # output tensor shape: (1, 3, 8, 8)

        x1_flat = self.flatten1(x1)  # output tensor shape: (1, 3072)
        x2_flat = self.flatten2(x2)  # output tensor shape: (1, 768)
        x3_flat = self.flatten1(x3)  # output tensor shape: (1, 192)

        x_concat = paddle.concat([x1_flat, x2_flat, x3_flat], axis=1)  # output tensor shape: (1, 4032)

        x_flat = self.flatten2(x_concat)  # output tensor shape: (1, 4032)
        x = self.linear1(x_flat)
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)