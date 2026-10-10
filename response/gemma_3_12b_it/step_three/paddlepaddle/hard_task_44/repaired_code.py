import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1
        self.conv1_1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv1_3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_5 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.dropout = nn.Dropout(p=0.25)
        self.concat1 = paddle.concat([x1, x2, x3], axis=1)

        # Block 2
        self.branch1_conv1 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.branch2_conv1 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.branch2_conv5 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.branch3_pool = F.max_pool2d(x=None, kernel_size=3, stride=2, padding=1) # Dummy value, will be assigned later
        self.branch3_conv1 = nn.Conv2D(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.concat2 = paddle.concat()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=128 * 32 * 32, out_features=10) # Assuming input size 32x32 after pooling


    def forward(self, x):
        # Block 1
        x1 = self.conv1_1(x)
        x2 = self.conv1_3(x)
        x3 = self.conv1_5(x)
        x = paddle.concat([x1, x2, x3], axis=1)
        x = self.dropout(x)

        # Block 2
        x1 = self.branch1_conv1(x)
        x2 = self.branch2_conv1(x)
        x3 = self.branch2_conv5(x)
        x4 = self.branch3_pool(x)
        x4 = self.branch3_conv1(x4)
        x = paddle.concat([x1, x2, x3, x4], axis=1)

        x = self.flatten(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)