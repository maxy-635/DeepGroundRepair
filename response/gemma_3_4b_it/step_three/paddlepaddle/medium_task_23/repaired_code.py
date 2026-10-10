import paddle
import paddle.nn as nn
import paddle.io
from paddle.summary import summary

class DL_Model(paddle.nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1: Single 1x1 convolution
        self.conv1 = nn.Conv2D(channels=3,
                                 out_channels=64,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        # Path 2: 1x1, 1x7, 7x1 convolutions
        self.conv2 = nn.Conv2D(channels=64,
                                 out_channels=128,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.conv3 = nn.Conv2D(channels=128,
                                 out_channels=64,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.conv4 = nn.Conv2D(channels=64,
                                 out_channels=64,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        # Path 3: 1x1, two sets of 1x7 and 7x1 convolutions
        self.conv5 = nn.Conv2D(channels=64,
                                 out_channels=128,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.conv6 = nn.Conv2D(channels=128,
                                 out_channels=64,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.conv7 = nn.Conv2D(channels=64,
                                 out_channels=64,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.conv8 = nn.Conv2D(channels=64,
                                 out_channels=64,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)


        # Path 4: Average pooling, 1x1 convolution
        self.avg_pool = nn.AvgPool2D(pool_size=2, stride=2)
        self.conv9 = nn.Conv2D(channels=64,
                                 out_channels=64,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        # Concatenation
        self.concat = nn.Concat(dim=1)

        # Flatten and Fully Connected Layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=64 * 4,
                            out_features=10)  # 10 classes for CIFAR-10

    def forward(self, x):
        # Path 1
        path1 = self.conv1(x)

        # Path 2
        path2 = self.conv2(x)
        path2 = self.conv3(path2)
        path2 = self.conv4(path2)

        # Path 3
        path3 = self.conv5(x)
        path3 = self.conv6(path3)
        path3 = self.conv7(path3)
        path3 = self.conv8(path3)

        # Path 4
        path4 = self.avg_pool(x)
        path4 = self.conv9(path4)

        # Concatenate
        x = self.concat([path1, path2, path3, path4])

        # Flatten and Fully Connected
        x = self.flatten(x)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor for validation
    input_tensor = paddle.randn(batch_size=1, channels=3, height=32, width=32, dtype=paddle.float32)
    output_tensor = model(input_tensor)
    summary(model, input_size=(1, 3, 32, 32))