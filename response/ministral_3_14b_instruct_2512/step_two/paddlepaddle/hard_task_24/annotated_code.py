import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial 1x1 convolutional layer
        self.initial_conv = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Branch 1: 3x3 convolutional layer
        self.branch1_conv = nn.Conv2D(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Branch 2: Max pooling + 3x3 convolutional layer + Upsampling
        self.branch2_pool = nn.MaxPool2D(
            kernel_size=2,
            stride=2,
            padding=0
        )
        self.branch2_conv = nn.Conv2D(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.branch2_upsample = nn.Upsample(
            scale_factor=2,
            mode='nearest'
        )

        # Branch 3: Max pooling + 3x3 convolutional layer + Upsampling
        self.branch3_pool = nn.MaxPool2D(
            kernel_size=2,
            stride=2,
            padding=0
        )
        self.branch3_conv = nn.Conv2D(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.branch3_upsample = nn.Upsample(
            scale_factor=2,
            mode='nearest'
        )

        # Fusion layer: 1x1 convolutional layer
        self.fusion_conv = nn.Conv2D(
            in_channels=192,  # 64 (branch1) + 64 (branch2) + 64 (branch3)
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Fully connected layers
        self.fc1 = nn.Linear(
            in_features=64 * 32 * 32,  # Assuming input size is 32x32
            out_features=512
        )
        self.fc2 = nn.Linear(
            in_features=512,
            out_features=256
        )
        self.fc3 = nn.Linear(
            in_features=256,
            out_features=10
        )

    def forward(self, x):
        # Initial 1x1 convolution
        x = self.initial_conv(x)

        # Branch 1
        branch1 = self.branch1_conv(x)

        # Branch 2
        branch2 = self.branch2_pool(x)
        branch2 = self.branch2_conv(branch2)
        branch2 = self.branch2_upsample(branch2)

        # Branch 3
        branch3 = self.branch3_pool(x)
        branch3 = self.branch3_conv(branch3)
        branch3 = self.branch3_upsample(branch3)

        # Concatenate all branches
        x = paddle.concat([branch1, branch2, branch3], axis=1)

        # Fusion layer
        x = self.fusion_conv(x)

        # Flatten for fully connected layers
        x = paddle.flatten(x, start_axis=1)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)