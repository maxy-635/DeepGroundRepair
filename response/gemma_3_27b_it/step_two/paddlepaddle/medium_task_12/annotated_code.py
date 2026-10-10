import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1_1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2D(num_features=16)
        self.relu1_1 = nn.ReLU()

        # Block 2
        self.conv2_1 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2D(num_features=32)
        self.relu2_1 = nn.ReLU()

        # Block 3
        self.conv3_1 = nn.Conv2D(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')
        self.bn3_1 = nn.BatchNorm2D(num_features=64)
        self.relu3_1 = nn.ReLU()

        # Flatten and Fully Connected Layers
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=64 * 32 * 32, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=10)
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1
        x = self.conv1_1(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.bn1_1(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.relu1_1(x)  # output tensor shape: (1, 16, 32, 32)

        # Block 2
        x_2 = self.conv2_1(x)  # output tensor shape: (1, 32, 32, 32)
        x_2 = self.bn2_1(x_2)  # output tensor shape: (1, 32, 32, 32)
        x_2 = self.relu2_1(x_2)  # output tensor shape: (1, 32, 32, 32)
        x = paddle.concat([x, x_2], axis=1)  # output tensor shape: (1, 48, 32, 32)

        # Block 3
        x_3 = self.conv3_1(x)
        x_3 = self.bn3_1(x_3)
        x_3 = self.relu3_1(x_3)
        x = paddle.concat([x, x_3], axis=1)


        # Flatten and Fully Connected Layers
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.softmax(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Example input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)