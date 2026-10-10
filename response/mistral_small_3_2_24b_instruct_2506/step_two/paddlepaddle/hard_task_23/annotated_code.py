import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial 1x1 convolution
        self.initial_conv = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # First branch: local feature extraction with two 3x3 convolutions
        self.branch1 = nn.Sequential(
            nn.Conv2D(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.Conv2D(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        # Second branch: average pooling, 3x3 convolution, and transposed convolution
        self.branch2 = nn.Sequential(
            nn.AvgPool2D(
                kernel_size=2,
                stride=2,
                padding=0
            ),
            nn.Conv2D(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.Conv2DTranspose(
                in_channels=64,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0
            )
        )

        # Third branch: same as second branch
        self.branch3 = nn.Sequential(
            nn.AvgPool2D(
                kernel_size=2,
                stride=2,
                padding=0
            ),
            nn.Conv2D(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.Conv2DTranspose(
                in_channels=64,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0
            )
        )

        # 1x1 convolution after concatenation
        self.concat_conv = nn.Conv2D(
            in_channels=192,  # 64*3
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Fully connected layer for classification
        self.fc = nn.Linear(
            in_features=64 * 32 * 32,  # Assuming input size is 32x32
            out_features=10
        )

    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)

        # Process through each branch
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)

        # Concatenate the outputs
        concat_out = paddle.concat([branch1_out, branch2_out, branch3_out], axis=1)

        # 1x1 convolution after concatenation
        concat_out = self.concat_conv(concat_out)

        # Flatten and pass through fully connected layer
        flatten_out = paddle.flatten(concat_out, start_axis=1, stop_axis=-1)
        out = self.fc(flatten_out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected shape: [1, 10]