import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super().__init__()
        # Basic block
        self.conv_block1 = nn.Conv2D(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn1 = nn.BatchNorm2D(16)
        self.relu1 = nn.ReLU()

        self.conv_block2 = nn.Conv2D(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2D(16)
        self.relu2 = nn.ReLU()

        self.branch_conv = nn.Conv2D(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Main structure
        self.conv1 = nn.Conv2D(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn_main1 = nn.BatchNorm2D(16)
        self.relu_main1 = nn.ReLU()

        self.conv2 = nn.Conv2D(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn_main2 = nn.BatchNorm2D(16)
        self.relu_main2 = nn.ReLU()

        self.avg_pool = nn.AvgPool2D(
            kernel_size=2,
            stride=2,
            padding=0
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features='mask', out_features=10)




    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Main path
        main_path = self.relu_main1(self.bn_main1(self.conv1(x)))  # output tensor shape: (1, 16, 32, 32)
        main_path = self.relu_main2(self.bn_main2(self.conv2(main_path)))  # output tensor shape: (1, 16, 32, 32)

        # Branch path
        branch_path = self.branch_conv(x)  # output tensor shape: (1, 16, 32, 32)

        # Feature fusion
        merged_features = paddle.add(main_path, branch_path)  # output tensor shape: (1, 16, 32, 32)

        # Downsampling and classification
        avg_pooled = self.avg_pool(merged_features)  # output tensor shape: (1, 16, 16, 16)
        flattened = self.flatten(avg_pooled)  # output tensor shape: (1, 4096)
        output = self.linear(flattened)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.rand(shape=[1, 3, 32, 32])  # Dummy input tensor (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)