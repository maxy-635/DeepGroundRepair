import paddle
import paddle.vision.transforms as T
from paddle.io import DataLoader

class DL_Model(paddle.nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.avg_pool = paddle.nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=5,
            stride=3,
            padding=2,
            groups=1
        )
        self.conv1 = paddle.nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
        self.fc1 = paddle.nn.Linear(64 * 7 * 7, 128)
        self.fc2 = paddle.nn.Linear(128, 10)
        self.dropout = paddle.nn.Dropout(0.5)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = paddle.nn.functional.relu(x)
        x = paddle.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = paddle.nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(1, 1, 28, 28)
    output_tensor = model(input_tensor)
    print(output_tensor)