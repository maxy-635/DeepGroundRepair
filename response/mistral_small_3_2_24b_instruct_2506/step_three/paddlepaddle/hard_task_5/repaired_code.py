import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self, num_classes=10):
        super(DL_Model, self).__init__()

        # Block 1: Split and 1x1 convolutions
        self.block1_conv1 = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.block1_conv2 = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.block1_conv3 = nn.Conv2D(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0)

        # Block 2: Channel shuffling
        self.block2_conv = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Block 3: 3x3 depthwise separable convolution
        self.block3_depthwise = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, groups=3)
        self.block3_pointwise = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Branch connection
        self.branch_conv = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Final fully connected layer
        self.fc = nn.Linear(in_features=3*32*32, out_features=num_classes)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Block 1: Split and 1x1 convolutions
        x1 = self.block1_conv1(x)  # output tensor shape: (1, 1, 32, 32)
        x2 = self.block1_conv2(x)  # output tensor shape: (1, 1, 32, 32)
        x3 = self.block1_conv3(x)  # output tensor shape: (1, 1, 32, 32)
        x_block1 = paddle.concat([x1, x2, x3], axis=1)  # output tensor shape: (1, 3, 32, 32)

        # Block 2: Channel shuffling
        _, _, h, w = x_block1.shape  # output tensor shape: (1); (1); (1)
        x_block1_reshaped = paddle.reshape(x_block1, shape=[-1, 1, h, w])
        x_block1_permuted = paddle.transpose(x_block1_reshaped, perm=[0, 1, 3, 2, 4, 5])
        x_block2 = paddle.reshape(x_block1_permuted, shape=[-1, 3, h, w])

        # Block 3: 3x3 depthwise separable convolution
        x_block3 = self.block3_depthwise(x_block2)
        x_block3 = self.block3_pointwise(x_block3)

        # Branch connection
        x_branch = self.branch_conv(x)

        # Combine main path and branch
        x_combined = paddle.add(x_block3, x_branch)

        # Final fully connected layer
        x_flattened = paddle.flatten(x_combined, start_axis=1)
        output = self.fc(x_flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)