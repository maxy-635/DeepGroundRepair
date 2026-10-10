import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.concat = nn.Concat(dim='mask')

        # Block 2
        self.reshape1 = nn.Reshape(in_shape=(32, 32, 32))
        self.perm1 = nn.Permut(dim=[0, 1, 3, 2])
        self.reshape2 = nn.Reshape(in_shape=(32, 32, 32))
        self.fc = nn.Linear(in_features=32 * 32 * 32, out_features=10)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x1 = x[0]
        x2 = x[1]
        x = self.concat([x1, x2])

        # Block 2
        x = self.reshape1(x)
        x = self.perm1(x)
        x = self.reshape2(x)

        x = x.flatten()
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)