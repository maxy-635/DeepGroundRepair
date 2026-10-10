import sys
import paddle
from paddle.nn import Layer
from paddle import nn
import paddle.static
import numpy as np

class DL_Model(Layer):
    def __init__(self, num_classes=10):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.block1_conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.block1_conv3 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.block1_concat = nn.Concat(dim='mask')

        # Block 2
        self.block2_avgpool = nn.AvgPool2D(kernel_size=2, stride=2)
        self.block2_reshape = nn.Reshape(in_event_shape=(1, 1, 32, 32))
        self.block2_perm = nn.Permute([2, 3, 1, 0])
        self.block2_reshape2 = nn.Reshape(in_event_shape=(32, 32, 32, 1))

        # Block 3
        self.block3_conv = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Fully connected layer
        self.fc = paddle.static.nn.fc(in_size=64 * 32 * 32, out_size=num_classes, activation='softmax')

    def forward(self, x):
        # Block 1
        x1 = self.block1_conv1(x)
        x2 = self.block1_conv2(x1)
        x3 = self.block1_conv3(x2)
        x_block1 = self.block1_concat([x1, x2, x3])

        # Branch
        x_branch = self.block2_avgpool(x)
        x_branch = self.block2_reshape(x_branch)
        x_branch = self.block2_perm(x_branch)
        x_branch = self.block2_reshape2(x_branch)

        # Concatenate
        x = paddle.concat([x_block1, x_branch], dim=1)

        # Block 3
        x = self.block3_conv(x)

        # Fully connected layer
        x = x.reshape([x.shape[0], -1])
        output = self.fc(x)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor
    input_tensor = paddle.ones([1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor)