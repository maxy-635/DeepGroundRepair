import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First Block
        self.block1_group1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0),
        )
        self.block1_group2 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
        )
        self.block1_group3 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
        )

        # Second Block
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2D(kernel_size=2, stride=2, padding=0),
            nn.Conv2D(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
        )

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=64 * 32 * 32, out_features=10)  # Assuming input size 32x32 after convolutions

    def forward(self, x):
        # First Block
        x1 = self.block1_group1(x)
        x2 = self.block1_group2(x)
        x3 = self.block1_group3(x)
        merged_block1 = paddle.concat([x1, x2, x3], axis=-1)

        # Second Block
        branch1_out = self.branch1(merged_block1)
        branch2_out = self.branch2(merged_block1)
        branch3_out = self.branch3(merged_block1)
        branch4_out = self.branch4(merged_block1)

        merged_block2 = paddle.concat([branch1_out, branch2_out, branch3_out, branch4_out], axis=-1)

        # Flatten and Linear
        x = self.flatten(merged_block2)
        x = self.linear(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)