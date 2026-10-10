import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path block
        self.main_path_conv = nn.Conv2D(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Branch path
        self.branch_conv = nn.Conv2D(in_channels=1, out_channels='mask', kernel_size=3, stride=1, padding=1)







        # Three blocks in main path
        self.block1 = nn.Sequential(
            nn.Conv2D(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=32  # Separable convolution
            ),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2D(
                in_channels=64,  # 32 (original) + 32 (concatenated)
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=32
            ),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2D(
                in_channels=96,  # 64 + 32
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=32
            ),
            nn.ReLU()
        )

        # Final layers
        self.final_conv = nn.Conv2D(
            in_channels=128,  # 96 + 32
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=32 * 28 * 28, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 1, 28, 28)
        # Main path
        main_out = self.main_path_conv(x)  # output tensor shape: (1, 32, 28, 28)

        # First block
        block1_out = self.block1(main_out)  # output tensor shape: (1, 32, 28, 28)
        concat1 = paddle.concat([main_out, block1_out], axis=1)  # output tensor shape: (1, 64, 28, 28)

        # Second block
        block2_out = self.block2(concat1)  # output tensor shape: (1, 32, 28, 28)
        concat2 = paddle.concat([concat1, block2_out], axis=1)  # output tensor shape: (1, 96, 28, 28)

        # Third block
        block3_out = self.block3(concat2)  # output tensor shape: (1, 32, 28, 28)
        concat3 = paddle.concat([concat2, block3_out], axis=1)  # output tensor shape: (1, 128, 28, 28)

        # Branch path
        branch_out = self.branch_conv(x)  # output tensor shape of this layer should be (1, 128, 28, 28)

        # Fusion
        fused = paddle.add(concat3, branch_out)

        # Final processing
        final_out = self.final_conv(fused)
        flattened = self.flatten(final_out)
        output = self.fc(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 1, 28, 28))  # Batch of 1, 1 channel, 28x28 MNIST image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)