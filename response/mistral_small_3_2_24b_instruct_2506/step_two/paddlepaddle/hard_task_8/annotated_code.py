import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Primary path
        self.conv1x1_1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.depthwise_conv_3x3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32)
        self.conv1x1_2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Block 1: Branch path
        self.branch_depthwise_conv_3x3 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, groups=1)
        self.branch_conv1x1 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Block 2: Channel shuffling
        self.fc = nn.Linear(in_features='mask', out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Block 1: Primary path
        x_primary = self.conv1x1_1(x)  # output tensor shape: (1, 32, 28, 28)
        x_primary = F.relu(x_primary)  # output tensor shape: (1, 32, 28, 28)
        x_primary = self.depthwise_conv_3x3(x_primary)  # output tensor shape: (1, 32, 28, 28)
        x_primary = F.relu(x_primary)  # output tensor shape: (1, 32, 28, 28)
        x_primary = self.conv1x1_2(x_primary)  # output tensor shape: (1, 64, 28, 28)
        x_primary = F.relu(x_primary)  # output tensor shape: (1, 64, 28, 28)

        # Block 1: Branch path
        x_branch = self.branch_depthwise_conv_3x3(x)  # output tensor shape: (1, 32, 28, 28)
        x_branch = F.relu(x_branch)  # output tensor shape: (1, 32, 28, 28)
        x_branch = self.branch_conv1x1(x_branch)  # output tensor shape: (1, 64, 28, 28)
        x_branch = F.relu(x_branch)  # output tensor shape: (1, 64, 28, 28)

        # Concatenate along channel dimension
        x = paddle.concat([x_primary, x_branch], axis=1)  # output tensor shape: (1, 128, 28, 28)

        # Block 2: Channel shuffling
        _, _, h, w = x.shape  # output tensor shape: (1); (1); (1)
        x = paddle.reshape(x, shape=[-1, h, w, 128])  # output tensor shape: (1, 28, 28, 128)
        x = paddle.transpose(x, perm=[0, 3, 1, 2])  # output tensor shape: (1, 128, 28, 28)
        x = paddle.reshape(x, shape=[-1, 128, h, w])  # output tensor shape: (1, 128, 28, 28)

        # Flatten and fully connected layer
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)  # output tensor shape: (1, 100352)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 1, 28, 28])  # MNIST image size is 28x28
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]