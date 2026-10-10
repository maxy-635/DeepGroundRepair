import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Define the three channel groups
        self.group1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        )
        self.group2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        )
        self.group3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        )

        # Concatenate the outputs from the three groups
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features='mask', out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x1 = self.group1(x)  # output tensor shape: (1, 32, 32, 32)
        x2 = self.group2(x)  # output tensor shape: (1, 32, 32, 32)
        x3 = self.group3(x)  # output tensor shape: (1, 32, 32, 32)

        # Concatenate the outputs
        x = paddle.concat([x1, x2, x3], axis=1)  # output tensor shape: (1, 96, 32, 32)
        x = self.flatten(x)  # output tensor shape: (1, 98304)
        x = self.fc1(x)
        x = paddle.nn.ReLU()(x)
        x = self.fc2(x)
        x = paddle.nn.ReLU()(x)
        x = self.fc3(x)
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)