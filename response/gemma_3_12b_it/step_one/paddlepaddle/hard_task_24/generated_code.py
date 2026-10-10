import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Initial 1x1 convolutional layer
        self.initial_conv = nn.Conv2D(
            in_channels=3,
            out_channels=32,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Branch 1: Local features
        self.branch1 = nn.Sequential(
            nn.Conv2D(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        # Branch 2: Downsampling, Conv, Upsampling
        self.branch2 = nn.Sequential(
            nn.MaxPool2D(
                kernel_size=2,
                stride=2,
                padding=0
            ),
            nn.Conv2D(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.Upsample(
                size=None,
                scale_factor=2,
                mode="nearest",
                align_corners=False
            )
        )

        # Branch 3: Downsampling, Conv, Upsampling
        self.branch3 = nn.Sequential(
            nn.MaxPool2D(
                kernel_size=2,
                stride=2,
                padding=0
            ),
            nn.Conv2D(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.Upsample(
                size=None,
                scale_factor=2,
                mode="nearest",
                align_corners=False
            )
        )

        # Fusion layer
        self.fusion_conv = nn.Conv2D(
            in_channels=64 * 3,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Fully connected layers
        self.fc1 = nn.Linear(
            in_features=64 * 32 * 32,
            out_features=128
        )
        self.fc2 = nn.Linear(
            in_features=128,
            out_features=64
        )
        self.fc3 = nn.Linear(
            in_features=64,
            out_features=10
        )

    def forward(self, x):
        # Initial convolutional layer
        x = self.initial_conv(x)

        # Branches
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)

        # Concatenation
        merged = paddle.concat([branch1_out, branch2_out, branch3_out], axis=1)

        # Fusion convolutional layer
        merged = self.fusion_conv(merged)

        # Flatten
        merged = paddle.flatten(merged, start=1)

        # Fully connected layers
        x = self.fc1(merged)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)