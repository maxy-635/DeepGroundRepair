import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path
        self.main_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),

            # Branch 1
            nn.Sequential(
                nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
            ),

            # Branch 2
            nn.Sequential(
                nn.AvgPool2D(kernel_size=2, stride=2, padding=0),
                nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.Conv2DTranspose(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, output_padding=0)
            ),

            # Branch 3
            nn.Sequential(
                nn.AvgPool2D(kernel_size=2, stride=2, padding=0),
                nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.Conv2DTranspose(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, output_padding=0)
            )
        )

        # Concatenate and 1x1 conv
        self.concat_conv = nn.Sequential(
            nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Branch path
        self.branch_path = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Final layers
        self.fc = nn.Linear(in_features=64*32*32, out_features=10)

    def forward(self, x):
        # Main path processing
        main_out = self.main_path[0](x)  # 1x1 conv

        # Branch 1
        branch1 = self.main_path[1](main_out)

        # Branch 2
        branch2 = self.main_path[2](main_out)

        # Branch 3
        branch3 = self.main_path[3](main_out)

        # Concatenate all branches
        concat = paddle.concat([branch1, branch2, branch3], axis=1)

        # 1x1 conv after concatenation
        main_out = self.concat_conv(concat)

        # Branch path processing
        branch_out = self.branch_path(x)

        # Fusion
        fused = main_out + branch_out

        # Flatten and final classification
        fused = paddle.flatten(fused, start_axis=1)
        out = self.fc(fused)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # Should be [1, 10]