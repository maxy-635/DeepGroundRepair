import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AvgPool2D(kernel_size=5, stride=3, padding=0)
        self.conv = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32*5*5, out_features=128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)