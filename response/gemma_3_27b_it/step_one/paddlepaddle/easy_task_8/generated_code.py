import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.depthwise_conv = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, groups=1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.25)
        self.pointwise_conv = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.25)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=64 * 28 * 28, out_features=10)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.pointwise_conv(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)