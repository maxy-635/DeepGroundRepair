import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.block1_conv1 = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.block1_conv2 = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.block1_conv3 = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.block1_relu = nn.ReLU()

        # Block 2
        self.block2_shape = paddle.shape(x='mask')
        self.block2_perm = nn.Permutation(dims=[0, 2, 3, 1])

        # Block 3
        self.block3_conv = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, groups=3)

        # Branch path
        self.branch_avgpool = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.branch_flatten = nn.Flatten()
        self.branch_linear = nn.Linear(in_features=3 * 3 * 3, out_features=10)  # Assuming CIFAR-10

        # Main path
        self.main_linear = nn.Linear(in_features=3 * 3 * 3, out_features=10)  # Assuming CIFAR-10

    def forward(self, x):
        # Block 1
        x1 = self.block1_relu(self.block1_conv1(x))
        x2 = self.block1_relu(self.block1_conv2(x))
        x3 = self.block1_relu(self.block1_conv3(x))
        block1_output = paddle.concat([x1, x2, x3], axis=1)

        # Block 2
        block2_shape = paddle.shape(x=block1_output)
        batch_size, channels, height, width = block2_shape
        block2_output = block1_output.reshape((batch_size, 3, height, width))
        block2_output = self.block2_perm(block2_output)
        block2_output = block2_output.reshape((batch_size, channels, height, width))

        # Block 3
        block3_output = self.block3_conv(block2_output)

        # Branch path
        branch_output = self.branch_avgpool(x)
        branch_output = self.branch_flatten(branch_output)
        branch_output = self.branch_linear(branch_output)

        # Main path
        main_output = block3_output.reshape([block2_shape[0], -1])
        main_output = self.main_linear(main_output)

        # Concatenate and output
        combined_output = paddle.concat([main_output, branch_output], axis=1)
        return combined_output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # Example input tensor for CIFAR-10
    output_tensor = model(input_tensor)
    print(output_tensor.shape)