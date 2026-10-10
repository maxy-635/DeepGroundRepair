import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2D(3, 64, kernel_size=1, stride=1, padding=0)
        self.pool1 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2D(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)