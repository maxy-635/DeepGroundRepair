import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2D(1, 16, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv2D(16, 32, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(0.2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=25088, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        x = self.conv1(x)  # output tensor shape: (1, 16, 28, 28)
        x = self.dropout1(x)  # output tensor shape: (1, 16, 28, 28)
        x = self.conv2(x)  # output tensor shape: (1, 32, 28, 28)
        x = self.dropout2(x)  # output tensor shape: (1, 32, 28, 28)
        x = self.flatten(x)  # output tensor shape: (1, 25088)
        x = self.linear(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 1, 28, 28))
    output_tensor = model(input_tensor)