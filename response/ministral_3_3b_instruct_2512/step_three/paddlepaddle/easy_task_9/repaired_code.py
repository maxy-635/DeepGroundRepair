import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1)
        self.depthwise_conv = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, depthwise=True, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=1, kernel_size=1, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=32*7*7, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.depthwise_conv(x)
        x = self.conv2(x)
        x = paddle.add(x, x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.to_tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # Example input with shape (1, 1, 2, 2)
    output_tensor = model(input_tensor)
    print(output_tensor)