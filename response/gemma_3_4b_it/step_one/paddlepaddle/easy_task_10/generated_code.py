import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=32 * 7 * 7, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(1, 1, 28, 28)  # Batch size 1, 1 channel, 28x28 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)