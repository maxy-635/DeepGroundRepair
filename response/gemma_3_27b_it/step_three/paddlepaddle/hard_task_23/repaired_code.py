import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        self.conv1x1_initial = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU()

        # Branch 1: Local Feature Extraction
        self.conv3x3_1 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3x3_2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        # Branch 2: Downsampling and Upsampling
        self.avgpool1 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.conv3x3_4 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.convtranspose1 = nn.Conv2DTranspose(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.relu5 = nn.ReLU()

        # Branch 3: Downsampling and Upsampling
        self.avgpool2 = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)
        self.conv3x3_5 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()
        self.convtranspose2 = nn.Conv2DTranspose(in_channels=32, out_channels=16, kernel_size=2, stride=2, padding=0)
        self.relu7 = nn.ReLU()

        # Consolidation
        self.conv1x1_final = nn.Conv2D(in_channels=64, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.relu8 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=64 * 8 * 8, out_features=10) # Assuming input image size is 32x32

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        x = self.conv1x1_initial(x)  # output tensor shape: (1, 16, 32, 32)
        x = self.relu1(x)  # output tensor shape: (1, 16, 32, 32)

        # Branch 1
        branch1 = self.conv3x3_1(x)  # output tensor shape: (1, 32, 32, 32)
        branch1 = self.relu2(branch1)  # output tensor shape: (1, 32, 32, 32)
        branch1 = self.conv3x3_2(branch1)  # output tensor shape: (1, 32, 32, 32)
        branch1 = self.relu3(branch1)  # output tensor shape: (1, 32, 32, 32)

        # Branch 2
        branch2 = self.avgpool1(x)  # output tensor shape: (1, 16, 16, 16)
        branch2 = self.conv3x3_4(branch2)  # output tensor shape: (1, 32, 16, 16)
        branch2 = self.relu4(branch2)  # output tensor shape: (1, 32, 16, 16)
        branch2 = self.convtranspose1(branch2)  # output tensor shape: (1, 16, 32, 32)
        branch2 = self.relu5(branch2)  # output tensor shape: (1, 16, 32, 32)

        # Branch 3
        branch3 = self.avgpool2(x)  # output tensor shape: (1, 16, 16, 16)
        branch3 = self.conv3x3_5(branch3)  # output tensor shape: (1, 32, 16, 16)
        branch3 = self.relu6(branch3)  # output tensor shape: (1, 32, 16, 16)
        branch3 = self.convtranspose2(branch3)  # output tensor shape: (1, 16, 32, 32)
        branch3 = self.relu7(branch3)  # output tensor shape: (1, 16, 32, 32)

        # Concatenation
        x = paddle.concat([branch1, branch2, branch3], axis=1)  # output tensor shape: (1, 64, 32, 32)

        # Consolidation
        x = self.conv1x1_final(x)
        x = self.relu8(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape) # Expected output shape: [1, 10]