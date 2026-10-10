import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1x1 = nn.Conv2D(3, 64, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2D(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2D(3, 64, kernel_size=5, stride=1, padding=2)
        self.conv_branch = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x1 = self.conv1x1(x)  # output tensor shape: (1, 64, 32, 32)
        x2 = self.conv3x3(x)  # output tensor shape: (1, 64, 32, 32)
        x3 = self.conv5x5(x)  # output tensor shape: (1, 64, 32, 32)
        x = paddle.concat([x1, x2, x3], axis=1)  # output tensor shape: (1, 192, 32, 32)
        x_branch = self.conv_branch(x)
        x = paddle.add(x, x_branch)
        x = paddle.flatten(x)
        x = self.fc1(x)
        x = paddle.nn.functional.relu(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)