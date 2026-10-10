import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1: Separable Convolution Groups
        self.sep_conv1_1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.sep_conv1_3 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.sep_conv1_5 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)

        # Block 2: Enhanced Feature Extraction
        self.conv2_3x3 = nn.Conv2D(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_1x1 = nn.Conv2D(in_channels=96, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv2_3x3_1 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_3x3_2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)

        # Global Average Pooling and Fully Connected Layer
        self.global_avg_pool = nn.AvgPool2D(kernel_size=8, stride=8, padding=0)
        self.fc = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # Block 1
        group1 = self.sep_conv1_1(x)
        group2 = self.sep_conv1_3(x)
        group3 = self.sep_conv1_5(x)
        block1_output = paddle.concat([group1, group2, group3], axis=1)  # Concatenate along the channel dimension

        # Block 2
        branch1 = self.conv2_3x3(block1_output)
        branch2 = self.conv2_1x1(block1_output)
        branch2 = self.conv2_3x3_1(F.relu(branch2))
        branch2 = self.conv2_3x3_2(F.relu(branch2))
        branch3 = self.max_pool(block1_output)

        block2_output = paddle.concat([branch1, branch2, branch3], axis=1)

        # Global Average Pooling and Fully Connected Layer
        gap = self.global_avg_pool(block2_output)
        gap = paddle.reshape(gap, [gap.shape[0], -1])
        output = self.fc(gap)
        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Example input tensor with shape (batch_size, channels, height, width)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)