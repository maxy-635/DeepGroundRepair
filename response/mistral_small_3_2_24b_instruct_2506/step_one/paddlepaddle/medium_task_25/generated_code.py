import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1: Single 1x1 convolution
        self.path1_conv = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Path 2: Average pooling followed by 1x1 convolution
        self.path2_pool = nn.AvgPool2D(
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.path2_conv = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Path 3: 1x1 convolution followed by parallel 1x3 and 3x1 convolutions
        self.path3_conv1 = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.path3_conv1x3 = nn.Conv2D(
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1)
        )
        self.path3_conv3x1 = nn.Conv2D(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0)
        )

        # Path 4: 1x1 convolution followed by 3x3 convolution, then parallel 1x3 and 3x1 convolutions
        self.path4_conv1 = nn.Conv2D(
            in_channels=3,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.path4_conv3x3 = nn.Conv2D(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.path4_conv1x3 = nn.Conv2D(
            in_channels=64,
            out_channels=64,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1)
        )
        self.path4_conv3x1 = nn.Conv2D(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0)
        )

        # Fully connected layer for classification
        self.fc = nn.Linear(
            in_features=64 * 32 * 32 * 4,  # 4 paths, each with 64 channels
            out_features=10  # CIFAR-10 has 10 classes
        )

    def forward(self, x):
        # Path 1
        path1 = self.path1_conv(x)

        # Path 2
        path2 = self.path2_pool(x)
        path2 = self.path2_conv(path2)

        # Path 3
        path3 = self.path3_conv1(x)
        path3_1x3 = self.path3_conv1x3(path3)
        path3_3x1 = self.path3_conv3x1(path3)
        path3 = paddle.concat([path3_1x3, path3_3x1], axis=1)

        # Path 4
        path4 = self.path4_conv1(x)
        path4 = self.path4_conv3x3(path4)
        path4_1x3 = self.path4_conv1x3(path4)
        path4_3x1 = self.path4_conv3x1(path4)
        path4 = paddle.concat([path4_1x3, path4_3x1], axis=1)

        # Concatenate all paths
        features = paddle.concat([path1, path2, path3, path4], axis=1)

        # Flatten and classify
        features = paddle.flatten(features, start_axis=1)
        output = self.fc(features)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # batch_size=1, channels=3, height=32, width=32
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)