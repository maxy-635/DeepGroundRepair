import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=64),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2D(num_features=128),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2D(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask'),
            nn.BatchNorm2D(num_features=256),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=256 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x = self.block1(x)  # output tensor shape: (1, 64, 32, 32)
        x1 = x  # output tensor shape: (1, 64, 32, 32)
        x = self.block2(x)  # output tensor shape: (1, 128, 32, 32)
        x = paddle.concat([x1, x], axis=1)  # output tensor shape: (1, 192, 32, 32)
        x2 = x  # output tensor shape: (1, 192, 32, 32)
        x = self.block3(x)
        x = paddle.concat([x2, x], axis=1)
        x = self.flatten(x)
        x = self.fc1(x)
        x = paddle.nn.ReLU()(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.rand(shape=(1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)