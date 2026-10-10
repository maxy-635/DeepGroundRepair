import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_conv1 = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.block1_conv2 = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.block1_conv3 = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

        # Block 2
        # Channel shuffling will be implemented in forward

        # Block 3
        self.block3_dwconv = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0, groups=3)
        self.block3_relu = nn.ReLU()

        # Branch
        self.branch_conv = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.branch_relu = nn.ReLU()

        # Final layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=3 * 32 * 32, out_features=10)  # Assuming input size is 32x32

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1
        x1 = self.block1_conv1(x)  # output tensor shape: (1, 1, 32, 32)
        x2 = self.block1_conv2(x)  # output tensor shape: (1, 1, 32, 32)
        x3 = self.block1_conv3(x)  # output tensor shape: (1, 1, 32, 32)
        x = paddle.concat([x1, x2, x3], axis=1)  # output tensor shape: (1, 3, 32, 32)
        x = self.relu(x)  # output tensor shape: (1, 3, 32, 32)

        # Block 2
        shape = paddle.shape(x)  # output tensor shape: (4)
        height, width, channels = shape[2], shape[3], shape[1]  # output tensor shape: (1); (1); (1)
        groups = 3  # output tensor shape: (1)
        channels_per_group = channels // groups  # output tensor shape: (1)

        x = paddle.reshape_(x, [shape[0], height, width, groups, channels_per_group])  # output tensor shape: (1, 32, 32, 3, 1)
        x = paddle.transpose(x, perm=[0, 2, 3, 4, 1])  # output tensor shape: (1, 32, 3, 1, 32)
        x = paddle.reshape_(x, [shape[0], height, width, channels])  # output tensor shape: (1, 32, 32, 3)

        # Block 3
        x = self.block3_dwconv(x)
        x = self.block3_relu(x)

        # Branch
        branch = self.branch_conv(x)
        branch = self.branch_relu(branch)

        # Combine main path and branch
        x = x + branch

        # Final layer
        x = self.flatten(x)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Example input tensor
    output_tensor = model(input_tensor)
    print(output_tensor.shape)