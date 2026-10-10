import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class FeatureExtractionBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2D(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')






        self.relu = nn.ReLU()

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        identity = x  # output tensor shape: (1, 33, 28, 28)
        out = self.conv(x)  # output tensor shape: (1, 32, 28, 28)
        out = self.relu(out)  # output tensor shape: (1, 32, 28, 28)
        out = paddle.concat([identity, out], axis=1)  # output tensor shape: (1, 33, 28, 28)
        return out

class DL_Model(nn.Layer):
    def __init__(self):
        super().__init__()
        self.main_path_block1 = FeatureExtractionBlock(1, 32)
        self.main_path_block2 = FeatureExtractionBlock(32, 64)
        self.main_path_block3 = FeatureExtractionBlock(64, 128)
        self.branch_path_conv = nn.Conv2D(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(
            in_features=128 * 7 * 7,
            out_features=10
        )

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Main path
        out = self.main_path_block1(x)  # output tensor shape: (1, 33, 28, 28)
        out = self.main_path_block2(out)
        out = self.main_path_block3(out)

        # Branch path
        branch_out = self.branch_path_conv(x)

        # Feature fusion
        fused_out = out + branch_out

        # Flatten and linear
        flattened_out = self.flatten(fused_out)
        output = self.linear(flattened_out)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=(1, 1, 28, 28))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)