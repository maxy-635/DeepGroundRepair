import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main path 1x1 convolution
        self.main_conv1x1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Branch 1: 3x3 convolution
        self.branch1_conv3x3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Branch 2: average pooling + 3x3 convolution + transpose convolution
        self.branch2_avgpool = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.branch2_conv3x3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch2_upconv = nn.Conv2DTranspose(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)

        # Branch 3: average pooling + 3x3 convolution + transpose convolution
        self.branch3_avgpool = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.branch3_conv3x3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch3_upconv = nn.Conv2DTranspose(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)

        # Concatenation and 1x1 convolution
        self.main_conv1x1_after = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Branch path 1x1 convolution
        self.branch_conv1x1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Final fully connected layer
        self.fc = nn.Linear(in_features=64 * 32 * 32, out_features=10)

    def forward(self, x):
        # Main path
        main = self.main_conv1x1(x)

        # Branch 1
        branch1 = self.branch1_conv3x3(main)

        # Branch 2
        branch2 = self.branch2_avgpool(main)
        branch2 = self.branch2_conv3x3(branch2)
        branch2 = self.branch2_upconv(branch2)

        # Branch 3
        branch3 = self.branch3_avgpool(main)
        branch3 = self.branch3_conv3x3(branch3)
        branch3 = self.branch3_upconv(branch3)

        # Concatenate branches
        concat = paddle.concat([branch1, branch2, branch3], axis=1)
        main = self.main_conv1x1_after(concat)

        # Branch path
        branch = self.branch_conv1x1(x)

        # Fusion
        out = main + branch

        # Flatten and classify
        out = paddle.flatten(out, start_axis=1)
        out = self.fc(out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # CIFAR-10 input shape: (batch_size, 3, 32, 32)
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)