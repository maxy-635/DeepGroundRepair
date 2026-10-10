import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv = nn.Conv2D(in_channels=3, out_channels=7, kernel_size=7, stride=1, padding=3)
        self.ln = nn.LayerNorm(normalized_shape='mask')
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=7*7*7, out_features=7)
        self.fc2 = nn.Linear(in_features=7, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x = self.conv(x)  # output tensor shape: (1, 7, 32, 32)
        x = self.ln(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)