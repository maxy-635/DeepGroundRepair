import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.avg_pool1 = nn.AvgPool2D(kernel_size=1, stride=1)
        self.avg_pool2 = nn.AvgPool2D(kernel_size=2, stride=2)
        self.avg_pool3 = nn.AvgPool2D(kernel_size=4, stride=4)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(144, 10)

    def forward(self, x):
        x1 = self.avg_pool1(x)
        x2 = self.avg_pool2(x)
        x3 = self.avg_pool3(x)
        x = self.flatten(x1) + self.flatten(x2) + self.flatten(x3)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)