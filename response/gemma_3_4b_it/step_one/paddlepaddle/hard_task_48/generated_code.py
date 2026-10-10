import sys
import paddle
from paddle.nn import Conv2D, BatchNorm2D, ReLU, AvgPool2D, AdaptiveAvgPool2D, Flatten, Linear
from paddle.nn import LayerNorm

class DL_Model(nn.Layer):
    def __init__(self):
        super().__init__()

        # Block 1
        self.conv1_1 = Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = BatchNorm2D(num_features=64)
        self.relu1_1 = ReLU()
        self.conv1_2 = Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = BatchNorm2D(num_features=64)
        self.relu1_2 = ReLU()
        self.conv1_3 = Conv2D(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn1_3 = BatchNorm2D(num_features=64)
        self.relu1_3 = ReLU()
        self.concat1 = paddle.concat([self.relu1_1, self.relu1_2, self.relu1_3], axis=1)

        # Block 2
        self.conv2_1 = Conv2D(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.bn2_1 = BatchNorm2D(num_features=128)
        self.relu2_1 = ReLU()
        self.avgpool2_1 = AvgPool2D(kernel_size=3, stride=1, padding=1)
        self.conv2_2 = Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = BatchNorm2D(num_features=128)
        self.relu2_2 = ReLU()

        self.path1 = Conv2D(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.bnpath1 = BatchNorm2D(num_features=128)
        self.relupath1 = ReLU()

        self.path2 = Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bnpath2 = BatchNorm2D(num_features=128)
        self.relupath2 = ReLU()

        self.path3_a = Conv2D(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.bnpath3_a = BatchNorm2D(num_features=128)
        self.relupath3_a = ReLU()
        self.path3_b = Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bnpath3_b = BatchNorm2D(num_features=128)
        self.relupath3_b = ReLU()
        self.concat2 = paddle.concat([self.relupath1, self.relupath3_a, self.relupath3_b], axis=1)

        self.path4 = Conv2D(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.bnpath4 = BatchNorm2D(num_features=128)
        self.relupath4 = ReLU()
        self.conv4_1 = Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4_1 = BatchNorm2D(num_features=128)
        self.relu4_1 = ReLU()
        self.path4_a = Conv2D(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.bnpath4_a = BatchNorm2D(num_features=128)
        self.relupath4_a = ReLU()
        self.path4_b = Conv2D(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bnpath4_b = BatchNorm2D(num_features=128)
        self.relupath4_b = ReLU()
        self.concat4 = paddle.concat([self.relupath4, self.relupath4_a, self.relupath4_b], axis=1)

        self.concat_block2 = paddle.concat([self.concat1, self.concat2, self.concat4], axis=1)

        self.flatten = Flatten()
        self.linear = Linear(in_features=concat_block2.shape[-1], out_features=10)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu1_3(x)
        x = self.concat1

        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu2_1(x)
        x = self.avgpool2_1(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)

        x = self.path1(x)
        x = self.bnpath1(x)
        x = self.relupath1(x)

        x = self.path2(x)
        x = self.bnpath2(x)
        x = self.relupath2(x)

        x = self.path3_a(x)
        x = self.bnpath3_a(x)
        x = self.relupath3_a(x)

        x = self.path3_b(x)
        x = self.bnpath3_b(x)
        x = self.relupath3_b(x)

        x = self.path4(x)
        x = self.bnpath4(x)
        x = self.relupath4(x)

        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu4_1(x)

        x = self.path4_a(x)
        x = self.bnpath4_a(x)
        x = self.relupath4_a(x)

        x = self.path4_b(x)
        x = self.bnpath4_b(x)
        x = self.relupath4_b(x)

        x = paddle.concat([x, self.concat_block2], axis=1)
        x = self.flatten(x)
        x = self.linear(x)
        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)