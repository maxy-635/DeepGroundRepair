import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, paddle.nn.Layer).__init__()
        self.conv1 = nn.Conv2D(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = paddle.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = paddle.nn.functional.relu(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)