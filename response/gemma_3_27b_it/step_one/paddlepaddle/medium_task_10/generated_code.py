import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Initial Convolution Layer
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2D(num_features=16)
        self.relu1 = nn.ReLU()

        # Level 1: Basic Block
        self.conv_level1_main = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn_level1_main = nn.BatchNorm2D(num_features=16)
        self.relu_level1_main = nn.ReLU()

        # Level 2: Two Residual Blocks
        self.conv_level2_branch1 = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn_level2_branch1 = nn.BatchNorm2D(num_features=16)
        self.relu_level2_branch1 = nn.ReLU()

        self.conv_level2_main1 = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn_level2_main1 = nn.BatchNorm2D(num_features=16)
        self.relu_level2_main1 = nn.ReLU()

        self.conv_level2_branch2 = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn_level2_branch2 = nn.BatchNorm2D(num_features=16)
        self.relu_level2_branch2 = nn.ReLU()

        self.conv_level2_main2 = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn_level2_main2 = nn.BatchNorm2D(num_features=16)
        self.relu_level2_main2 = nn.ReLU()

        # Level 3: Global Branch
        self.conv_level3_branch = nn.Conv2D(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn_level3_branch = nn.BatchNorm2D(num_features=16)
        self.relu_level3_branch = nn.ReLU()

        # Final Layers
        self.avg_pool = nn.AvgPool2D(kernel_size=8, stride=8, padding=0)
        self.linear = nn.Linear(in_features=16 * 8 * 8, out_features=10)

    def forward(self, x):
        # Initial Convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Level 1
        residual_level1 = x
        x = self.conv_level1_main(x)
        x = self.bn_level1_main(x)
        x = self.relu_level1_main(x)
        x = paddle.add(x, residual_level1)

        # Level 2
        residual_level2 = x
        x_branch1 = self.conv_level2_branch1(x)
        x_branch1 = self.bn_level2_branch1(x_branch1)
        x_branch1 = self.relu_level2_branch1(x_branch1)

        x = self.conv_level2_main1(x)
        x = self.bn_level2_main1(x)
        x = self.relu_level2_main1(x)
        x = paddle.add(x, x_branch1)

        x_branch2 = self.conv_level2_branch2(x)
        x_branch2 = self.bn_level2_branch2(x_branch2)
        x_branch2 = self.relu_level2_branch2(x_branch2)
        
        x = self.conv_level2_main2(x)
        x = self.bn_level2_main2(x)
        x = self.relu_level2_main2(x)
        x = paddle.add(x, x_branch2)


        # Level 3
        residual_level3 = x
        x_branch3 = self.conv_level3_branch(residual_level1)
        x_branch3 = self.bn_level3_branch(x_branch3)
        x_branch3 = self.relu_level3_branch(x_branch3)

        x = paddle.add(x, x_branch3)

        # Final Layers
        x = self.avg_pool(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.linear(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Batch size 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]