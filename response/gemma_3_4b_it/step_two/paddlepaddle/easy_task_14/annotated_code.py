import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.AdaptiveAvgPool2D(output_size=(4, 4))
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(1, 3, 32, 32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)