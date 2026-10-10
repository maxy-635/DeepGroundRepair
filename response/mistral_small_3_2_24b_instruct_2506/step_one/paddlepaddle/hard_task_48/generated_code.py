import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1x1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn1x1 = nn.BatchNorm2D(num_features=64)

        self.conv3x3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3x3 = nn.BatchNorm2D(num_features=64)

        self.conv5x5 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn5x5 = nn.BatchNorm2D(num_features=64)

        # Block 2
        self.path1_conv = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)

        self.path2_pool = nn.AvgPool2D(kernel_size=3, stride=1, padding=1)
        self.path2_conv = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)

        self.path3_conv1 = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.path3_conv1x3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.path3_conv3x1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=1, padding=(1, 0))

        self.path4_conv1 = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.path4_conv3x3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.path4_conv1x3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.path4_conv3x1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=1, padding=(1, 0))

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=256 * 16 * 16, out_features=10)  # Assuming input size is 32x32 and after pooling it's 16x16

    def forward(self, x):
        # Block 1
        x1 = self.conv1x1(x)
        x1 = self.bn1x1(x1)

        x2 = self.conv3x3(x)
        x2 = self.bn3x3(x2)

        x3 = self.conv5x5(x)
        x3 = self.bn5x5(x3)

        x = paddle.concat([x1, x2, x3], axis=1)

        # Block 2
        # Path 1
        p1 = self.path1_conv(x)

        # Path 2
        p2 = self.path2_pool(x)
        p2 = self.path2_conv(p2)

        # Path 3
        p3 = self.path3_conv1(x)
        p3_1 = self.path3_conv1x3(p3)
        p3_2 = self.path3_conv3x1(p3)
        p3 = paddle.concat([p3_1, p3_2], axis=1)

        # Path 4
        p4 = self.path4_conv1(x)
        p4 = self.path4_conv3x3(p4)
        p4_1 = self.path4_conv1x3(p4)
        p4_2 = self.path4_conv3x1(p4)
        p4 = paddle.concat([p4_1, p4_2], axis=1)

        x = paddle.concat([p1, p2, p3, p4], axis=1)

        # Final layers
        x = self.flatten(x)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]