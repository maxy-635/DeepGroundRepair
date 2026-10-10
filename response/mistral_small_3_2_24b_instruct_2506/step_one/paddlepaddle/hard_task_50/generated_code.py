import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block
        self.maxpool1 = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.maxpool4 = nn.MaxPool2D(kernel_size=4, stride=4, padding=0)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=3*3*3, out_features=128)
        self.reshape = paddle.reshape

        # Second block
        self.sepconv1 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0, groups=32)
        self.sepconv3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32)
        self.sepconv5 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2, groups=32)
        self.sepconv7 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3, groups=32)
        self.fc2 = nn.Linear(in_features=32*8*8, out_features=10)

    def forward(self, x):
        # First block
        x1 = self.maxpool1(x)
        x2 = self.maxpool2(x)
        x4 = self.maxpool4(x)

        x1 = paddle.flatten(x1)
        x2 = paddle.flatten(x2)
        x4 = paddle.flatten(x4)

        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        x4 = self.dropout(x4)

        x = paddle.concat([x1, x2, x4], axis=1)
        x = self.fc1(x)
        x = self.reshape(x, shape=[-1, 32, 8, 8])

        # Second block
        x = paddle.split(x, num_or_sections=4, axis=1)
        x1 = self.sepconv1(x[0])
        x2 = self.sepconv3(x[1])
        x3 = self.sepconv5(x[2])
        x4 = self.sepconv7(x[3])

        x = paddle.concat([x1, x2, x3, x4], axis=1)
        x = paddle.flatten(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)