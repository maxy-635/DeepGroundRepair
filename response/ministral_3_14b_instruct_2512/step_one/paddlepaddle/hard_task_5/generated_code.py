import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Split into 3 groups with 1x1 convolutions
        self.conv1_1 = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.conv1_3 = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Block 2: Channel shuffling
        self.conv2_1 = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

        # Block 3: Depthwise separable convolution
        self.depthwise_conv = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1, groups=3)
        self.pointwise_conv = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Branch connection
        self.branch_conv = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

        # Final fully connected layer
        self.fc = nn.Linear(in_features=3*32*32, out_features=10)  # Assuming input size is 32x32

        # Activation functions
        self.relu = nn.ReLU()
        self.prelu = nn.PReLU()

    def forward(self, x):
        # Block 1: Split and fuse
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x3 = self.conv1_3(x)
        x_block1 = paddle.concat([x1, x2, x3], axis=1)  # Concatenate along channel dimension

        # Block 2: Channel shuffling
        h, w, c = x_block1.shape[2], x_block1.shape[3], x_block1.shape[1]
        x_block2 = paddle.reshape(x_block1, [x_block1.shape[0], h, w, 3, c//3])
        x_block2 = paddle.transpose(x_block2, [0, 1, 2, 4, 3])  # Swap last two dimensions
        x_block2 = paddle.reshape(x_block2, [x_block2.shape[0], c, h, w])

        # Block 3: Depthwise separable convolution
        x_block3 = self.depthwise_conv(x_block2)
        x_block3 = self.pointwise_conv(x_block3)

        # Branch connection
        branch = self.branch_conv(x)
        x_combined = x_block3 + branch

        # Final processing
        x_flat = paddle.reshape(x_combined, [x_combined.shape[0], -1])
        x_out = self.fc(x_flat)

        return x_out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Batch of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)