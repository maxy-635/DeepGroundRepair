import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super().__init__()
        # Block 1
        self.conv1_1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.depthwise_conv1 = nn.DepthwiseConv2D(kernel_size='mask', stride='mask', padding='mask')
        self.conv1_2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.depthwise_conv2 = nn.DepthwiseConv2D(kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)

        # Block 2
        self.fc = nn.Linear(in_features=128 * 7 * 7, out_features=10)

    def forward(self, x):
        # Block 1
        x = self.conv1_1(x)
        x = self.depthwise_conv1(x)
        x = self.conv1_2(x)

        branch_out = self.depthwise_conv2(x)
        branch_out = self.conv2_2(branch_out)

        x = paddle.concat([x, branch_out], axis=1)

        # Block 2
        shape = paddle.shape(x)
        height, width, _, channels = shape[0], shape[1], shape[2], shape[3]
        x = x.reshape([shape[0], shape[1], 4, channels // 4])
        x = paddle.transpose(x, [0, 2, 1, 3])
        x = x.reshape([shape[0], shape[1], channels])

        x = x.reshape([shape[0], -1])
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 1, 28, 28])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)