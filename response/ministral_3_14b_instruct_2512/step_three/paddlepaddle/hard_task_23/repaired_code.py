import paddle
import paddle.nn as nn
import paddle.nn.functional as F

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

        # Local feature extraction branch
        self.local_branch = nn.Sequential(
            nn.Conv2D(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2D(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU()
        )

        # Second branch: average pooling + 3x3 conv + transposed conv
        self.pool1 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.conv_pool1 = nn.Conv2D(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.up1 = nn.Conv2DTranspose(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )

        # Third branch: average pooling + 3x3 conv + transposed conv
        self.pool2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.conv_pool2 = nn.Conv2D(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.up2 = nn.Conv2DTranspose(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )

        # 1x1 convolution after concatenation
        self.final_conv = nn.Conv2D(
            in_channels=192,  # 64 (local) + 64 (branch1) + 64 (branch2)
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

        # ReLU activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # Initial 1x1 convolution
        x = self.initial_conv(x)
        x = self.relu(x)

        # Local feature extraction branch
        local = self.local_branch(x)

        # Second branch
        branch1 = self.pool1(x)
        branch1 = self.conv_pool1(branch1)
        branch1 = self.relu(branch1)
        branch1 = self.up1(branch1)

        # Third branch
        branch2 = self.pool2(x)
        branch2 = self.conv_pool2(branch2)
        branch2 = self.relu(branch2)
        branch2 = self.up2(branch2)

        # Concatenate all branches
        x = paddle.concat([local, branch1, branch2], axis=1)

        # 1x1 convolution after concatenation
        x = self.final_conv(x)
        x = self.relu(x)

        # Flatten and fully connected layer
        x = paddle.flatten(x, start_axis=1)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)