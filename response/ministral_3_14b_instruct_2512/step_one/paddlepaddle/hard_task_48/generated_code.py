import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Three parallel branches with separable convolutions of different kernel sizes
        self.conv1x1_1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn1x1_1 = nn.BatchNorm2D(num_features=64)

        self.conv3x3_1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3x3_1 = nn.BatchNorm2D(num_features=64)

        self.conv5x5_1 = nn.Conv2D(in_channels=3, out_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn5x5_1 = nn.BatchNorm2D(num_features=64)

        # Block 2: Four parallel branches with different operations
        # Path 1: 1x1 convolution
        self.conv1x1_2 = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn1x1_2 = nn.BatchNorm2D(num_features=64)

        # Path 2: 3x3 average pooling followed by 1x1 convolution
        self.avgpool_2 = nn.AvgPool2D(kernel_size=3, stride=1, padding=1)
        self.conv1x1_3 = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn1x1_3 = nn.BatchNorm2D(num_features=64)

        # Path 3: 1x1 convolution followed by 1x3 and 3x1 convolutions
        self.conv1x1_4 = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn1x1_4 = nn.BatchNorm2D(num_features=64)
        self.conv1x3_4 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv3x1_4 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=(3, 1), stride=1, padding=(1, 0))

        # Path 4: 1x1 convolution followed by 3x3 convolution and then 1x3 and 3x1 convolutions
        self.conv1x1_5 = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn1x1_5 = nn.BatchNorm2D(num_features=64)
        self.conv3x3_5 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1x3_5 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.conv3x1_5 = nn.Conv2D(in_channels=64, out_channels=32, kernel_size=(3, 1), stride=1, padding=(1, 0))

        # Final classification layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=192 * 32 * 32, out_features=10)  # Assuming input size is 32x32

    def forward(self, x):
        # Block 1 processing
        x1 = self.bn1x1_1(self.conv1x1_1(x))
        x2 = self.bn3x3_1(self.conv3x3_1(x))
        x3 = self.bn5x5_1(self.conv5x5_1(x))

        # Concatenate the outputs of Block 1
        x = paddle.concat([x1, x2, x3], axis=1)

        # Block 2 processing
        # Path 1
        path1 = self.bn1x1_2(self.conv1x1_2(x))

        # Path 2
        path2 = self.bn1x1_3(self.conv1x1_3(self.avgpool_2(x)))

        # Path 3
        path3 = self.bn1x1_4(self.conv1x1_4(x))
        path3_1 = self.conv1x3_4(path3)
        path3_2 = self.conv3x1_4(path3)
        path3 = paddle.concat([path3_1, path3_2], axis=1)

        # Path 4
        path4 = self.bn1x1_5(self.conv1x1_5(x))
        path4 = self.conv3x3_5(path4)
        path4_1 = self.conv1x3_5(path4)
        path4_2 = self.conv3x1_5(path4)
        path4 = paddle.concat([path4_1, path4_2], axis=1)

        # Concatenate all paths
        x = paddle.concat([path1, path2, path3, path4], axis=1)

        # Final classification
        x = self.flatten(x)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)