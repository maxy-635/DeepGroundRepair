import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1: Single 1x1 convolution
        self.path1 = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Path 2: Average pooling followed by 1x1 convolution
        self.path2 = nn.Sequential(
            nn.AvgPool2D(kernel_size=3, stride=1, padding=1),
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

        # Path 3: 1x1 conv followed by parallel 1x3 and 3x1 convolutions
        self.path3_conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.path3_conv1x3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.path3_conv3x1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.path3_relu = nn.ReLU()

        # Path 4: 1x1 conv followed by 3x3 conv, then parallel 1x3 and 3x1 convolutions
        self.path4_conv1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.path4_conv3x3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.path4_conv1x3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.path4_conv3x1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.path4_relu = nn.ReLU()

        # Final concatenation and classification
        self.fc = nn.Linear(in_features=64*4, out_features=10)  # 4 paths * 64 channels

    def forward(self, x):
        # Path 1
        out1 = self.path1(x)

        # Path 2
        out2 = self.path2(x)

        # Path 3
        x3 = self.path3_conv1(x)
        x3_1 = self.path3_conv1x3(x3)
        x3_2 = self.path3_conv3x1(x3)
        out3 = self.path3_relu(paddle.concat([x3_1, x3_2], axis=1))

        # Path 4
        x4 = self.path4_conv1(x)
        x4 = self.path4_conv3x3(x4)
        x4_1 = self.path4_conv1x3(x4)
        x4_2 = self.path4_conv3x1(x4)
        out4 = self.path4_relu(paddle.concat([x4_1, x4_2], axis=1))

        # Concatenate all paths
        out = paddle.concat([out1, out2, out3, out4], axis=1)

        # Global average pooling to reduce spatial dimensions
        out = F.adaptive_avg_pool2d(out, (1, 1))

        # Flatten and classify
        out = paddle.flatten(out, start_axis=1)
        out = self.fc(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, channels, height, width]
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)