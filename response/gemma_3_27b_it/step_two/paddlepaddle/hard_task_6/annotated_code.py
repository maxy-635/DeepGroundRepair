import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_conv1_1 = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.block1_relu1_1 = nn.ReLU()
        self.block1_conv1_2 = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.block1_relu1_2 = nn.ReLU()
        self.block1_conv1_3 = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.block1_relu1_3 = nn.ReLU()

        # Block 2
        self.block2_depthwise_sep_conv = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, groups=3)
        self.block2_relu = nn.ReLU()

        # Block 3
        self.block3_depthwise_sep_conv = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, groups=3)
        self.block3_relu = nn.ReLU()

        # Branch Path
        self.branch_avgpool = nn.AvgPool2D(kernel_size=8, stride=8, padding=0)

        # Fully Connected Layer
        self.fc = nn.Linear(in_features=3*32*32, out_features=10)


    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main Path
        x1_1 = self.block1_conv1_1(x)  # output tensor shape: (1, 1, 32, 32)
        x1_1 = self.block1_relu1_1(x1_1)  # output tensor shape: (1, 1, 32, 32)
        x1_2 = self.block1_conv1_2(x)  # output tensor shape: (1, 1, 32, 32)
        x1_2 = self.block1_relu1_2(x1_2)  # output tensor shape: (1, 1, 32, 32)
        x1_3 = self.block1_conv1_3(x)  # output tensor shape: (1, 1, 32, 32)
        x1_3 = self.block1_relu1_3(x1_3)  # output tensor shape: (1, 1, 32, 32)
        x_main = paddle.concat([x1_1, x1_2, x1_3], axis=1)  # output tensor shape: (1, 3, 32, 32)

        # Block 2
        shape = x_main.shape  # output tensor shape: (unknown)
        x_block2 = paddle.reshape_(x_main, shape=[shape[0], shape[1], 3, shape[3]])
        x_block2 = paddle.transpose(x_block2, perm=[0, 2, 1, 3])
        x_block2 = paddle.reshape_(x_block2, shape=[shape[0], shape[1], shape[2], shape[3]])

        # Block 3
        x_block3 = self.block3_depthwise_sep_conv(x_block2)
        x_block3 = self.block3_relu(x_block3)


        # Branch Path
        x_branch = self.branch_avgpool(x)
        x_branch = paddle.reshape_(x_branch, shape=[x_branch.shape[0], -1])

        # Concatenation
        x_concat = paddle.concat([x_block3.reshape([x_block3.shape[0], -1]), x_branch], axis=1)

        # Fully Connected Layer
        x_output = self.fc(x_concat)

        return x_output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)