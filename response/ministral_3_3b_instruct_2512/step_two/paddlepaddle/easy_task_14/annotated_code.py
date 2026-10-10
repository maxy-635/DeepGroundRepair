import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.avg_pool = nn.AvgPool2D(kernel_size=1, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features='mask', out_features=10)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x = self.avg_pool(x)  # output tensor shape: (1, 3, 32, 32)
        x = x.flatten(start_dim=1)  # output tensor shape: (1, 3072)
        x = self.fc1(x)
        x = paddle.nn.functional.gelu(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)