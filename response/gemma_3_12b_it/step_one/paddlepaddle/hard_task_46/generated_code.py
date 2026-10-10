import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Separable Convolutions with different kernel sizes
        self.conv1_1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv1_3 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1_5 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)

        # Block 2: Multiple Branches
        self.conv2_1 = nn.Conv2D(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_2_1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv2_2_2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.max_pool = F.max_pool2d(input=None, kernel_size=3, stride=2, padding=1)

        self.global_avg_pool = F.adaptive_avg_pool2d(input=None, output_size=(1, 1))
        self.fc = nn.Linear(in_features=64, out_features=10)


    def forward(self, x):
        # Block 1
        x1_1 = self.conv1_1(x)
        x1_3 = self.conv1_3(x)
        x1_5 = self.conv1_5(x)
        x1 = paddle.concat([x1_1, x1_3, x1_5], axis=1)  # Concatenate along the channel dimension

        # Block 2
        x2_1 = self.conv2_1(x1)
        x2_2 = paddle.concat([self.conv2_2_1(x1), self.conv2_2_2(x1)], axis=1)
        x2 = paddle.concat([x2_1, x2_2], axis=1)

        x3 = self.max_pool(x1)

        x4 = paddle.concat([x2, x3], axis=1)

        # Global Average Pooling and Fully Connected Layer
        x5 = self.global_avg_pool(x4)
        x6 = self.fc(x5)

        return x6


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)