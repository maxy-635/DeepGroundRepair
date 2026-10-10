import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_primary_conv1x1 = nn.Conv2D(in_channels=1, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.block1_primary_dwconv3x3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, groups=64)
        self.block1_primary_conv1x1_2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

        self.block1_branch_dwconv3x3 = nn.Conv2D(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask', groups='mask')
        self.block1_branch_conv1x1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Block 2
        self.block2_linear = nn.Linear(in_features=64 * 64, out_features=10)


    def forward(self, x):
        # Block 1
        primary_path = self.block1_primary_conv1x1(x)
        primary_path = self.block1_primary_dwconv3x3(primary_path)
        primary_path = self.block1_primary_conv1x1_2(primary_path)

        branch_path = self.block1_branch_dwconv3x3(x)
        branch_path = self.block1_branch_conv1x1(branch_path)

        x = paddle.concat([primary_path, branch_path], axis=1)

        # Block 2
        shape = x.shape
        height, width = shape[2], shape[3]
        groups = 4
        channels_per_group = shape[1] // groups

        x = x.reshape(shape[0], height, width, groups, channels_per_group)
        x = x.transpose((0, 2, 3, 4, 1))
        x = x.reshape(shape[0], height, width, shape[1])

        # Fully connected layer
        x = x.reshape(shape[0], -1)
        x = self.block2_linear(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)