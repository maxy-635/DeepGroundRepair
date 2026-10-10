import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super().__init__()
        # Block 1
        self.conv1_group1 = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv1_group2 = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.conv1_group3 = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)

        # Block 2
        self.block2_reshape = nn.Layer()
        self.block2_permute = nn.Layer()
        self.block2_reshape_back = nn.Layer()

        # Block 3
        self.depthwise_conv = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, groups=3, bias=True)

        # Branch
        self.branch_conv = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)

        # Final Layer
        self.linear = nn.Linear(in_features=1 * 32 * 32, out_features=10)  # Assuming input size is 32x32

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1
        x1 = self.conv1_group1(x)  # output tensor shape: (1, 1, 32, 32)
        x2 = self.conv1_group2(x)  # output tensor shape: (1, 1, 32, 32)
        x3 = self.conv1_group3(x)  # output tensor shape: (1, 1, 32, 32)
        x_concat = paddle.concat([x1, x2, x3], axis=1)  # output tensor shape: (1, 3, 32, 32)

        # Block 2
        batch_size, channels, height, width = x_concat.shape  # output tensor shape: (1); (1); (1); (1)
        x_reshaped = paddle.reshape(x_concat, [batch_size, channels // 3, 3, height, width])  # output tensor shape: (1, 1, 3, 32, 32)
        x_permuted = paddle.transpose(x_reshaped, [0, 2, 3, 4, 1])  # output tensor shape: (1, 3, 32, 32, 1)
        x_reshaped_back = paddle.reshape(x_permuted, [batch_size, channels, height, width])  # output tensor shape: (1, 3, 32, 32)
        

        # Block 3
        x_block3 = self.depthwise_conv(x_reshaped_back)

        # Branch
        x_branch = self.branch_conv(x)

        # Combine main path and branch
        x_combined = x_block3 + x_branch

        # Flatten and pass through fully connected layer
        x_flattened = paddle.flatten(x_combined, [1, 2, 3])
        x_output = self.linear(x_flattened)

        return x_output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.rand([1, 3, 32, 32])  # Dummy input tensor (batch_size=1, channels=3, height=32, width=32)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]