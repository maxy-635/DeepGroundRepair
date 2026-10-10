import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.conv1_main = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv2_main = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3_main = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)

        # Branch path
        self.conv1_branch = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Fusion
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32 * 32 * 32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.relu = nn.ReLU()

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path
        x1 = self.conv1_main(x)  # output tensor shape: (1, 32, 32, 32)
        x1 = self.relu(x1)  # output tensor shape: (1, 32, 32, 32)
        x1 = self.conv2_main(x1)  # output tensor shape: (1, 32, 32, 32)
        x1 = self.relu(x1)  # output tensor shape: (1, 32, 32, 32)
        x1 = self.conv3_main(x1)  # output tensor shape: (1, 32, 32, 32)
        x1 = self.relu(x1)  # output tensor shape: (1, 32, 32, 32)

        # Branch path
        x2 = self.conv1_branch(x)  # output tensor shape: (1, 32, 32, 32)
        x2 = self.relu(x2)  # output tensor shape: (1, 32, 32, 32)

        # Fusion
        x = paddle.concat([x1, x2], axis=1)  # output tensor shape: (1, 64, 32, 32)
        x = self.flatten(x)  # output tensor shape: (1, 65536)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)