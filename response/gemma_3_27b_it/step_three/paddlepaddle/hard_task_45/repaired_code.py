import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Depthwise Separable Convolutions
        self.conv1_1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv1_3 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1_5 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)

        # Block 2: Multiple Branches
        self.branch1 = nn.Conv2D(in_channels=96, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.branch2_conv1 = nn.Conv2D(in_channels=96, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.branch2_conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch2_conv3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch3_conv1 = nn.Conv2D(in_channels=96, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.branch3_conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.branch4_pool = nn.MaxPool2D(kernel_size=1, stride=1, padding=0)
        self.branch4_conv1 = nn.Conv2D(in_channels=96, out_channels=32, kernel_size=1, stride=1, padding=0)


        # Flatten and Linear Layer
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=32 * 8 * 8, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1
        x1_1 = self.conv1_1(x)  # output tensor shape: (1, 32, 32, 32)
        x1_3 = self.conv1_3(x)  # output tensor shape: (1, 32, 32, 32)
        x1_5 = self.conv1_5(x)  # output tensor shape: (1, 32, 32, 32)
        x_block1 = paddle.concat([x1_1, x1_3, x1_5], axis=1)  # output tensor shape: (1, 96, 32, 32)

        # Block 2
        branch1_out = self.branch1(x_block1)  # output tensor shape: (1, 32, 32, 32)
        branch2_out = self.branch2_conv1(x_block1)  # output tensor shape: (1, 32, 32, 32)
        branch2_out = self.branch2_conv2(branch2_out)  # output tensor shape: (1, 32, 32, 32)
        branch2_out = self.branch2_conv3(branch2_out)  # output tensor shape: (1, 32, 32, 32)
        branch3_out = self.branch3_conv1(x_block1)  # output tensor shape: (1, 32, 32, 32)
        branch3_out = self.branch3_conv2(branch3_out)  # output tensor shape: (1, 32, 32, 32)
        branch4_out = self.branch4_pool(x_block1)  # output tensor shape of this layer should be (1, 96, 32, 32)
        branch4_out = self.branch4_conv1(branch4_out)

        x_block2 = paddle.concat([branch1_out, branch2_out, branch3_out, branch4_out], axis=1)

        # Flatten and Linear
        x_flatten = self.flatten(x_block2)
        x_out = self.linear(x_flatten)

        return x_out


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)