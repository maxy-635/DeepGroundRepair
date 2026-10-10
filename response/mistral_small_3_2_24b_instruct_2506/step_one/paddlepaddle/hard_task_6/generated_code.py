import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self, num_classes=10):
        super(DL_Model, self).__init__()

        # Block 1: Split into three groups, 1x1 conv, and concatenate
        self.block1_conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.block1_conv2 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.block1_conv3 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=1, stride=1, padding=0)

        # Block 2: Channel shuffling
        self.block2_permute = nn.LayerList([nn.Layer() for _ in range(3)])  # Placeholder for permutation

        # Block 3: 3x3 depthwise separable convolution
        self.block3_depthwise = nn.Conv2D(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, groups=96)
        self.block3_pointwise = nn.Conv2D(in_channels=96, out_channels=96, kernel_size=1, stride=1, padding=0)

        # Branch path: Average pooling
        self.branch_avgpool = nn.AvgPool2D(kernel_size=3, stride=1, padding=1)

        # Final fully connected layer
        self.fc = nn.Linear(in_features=96 * 32 * 32, out_features=num_classes)

    def forward(self, x):
        # Block 1: Split and process
        x1, x2, x3 = paddle.split(x, num_or_sections=3, axis=1)
        x1 = F.relu(self.block1_conv1(x1))
        x2 = F.relu(self.block1_conv2(x2))
        x3 = F.relu(self.block1_conv3(x3))
        x = paddle.concat([x1, x2, x3], axis=1)

        # Block 2: Channel shuffling
        _, _, h, w = x.shape
        x = paddle.reshape(x, shape=[-1, 3, 32, h, w])
        x = paddle.transpose(x, perm=[0, 2, 4, 1, 3])
        x = paddle.reshape(x, shape=[-1, 96, h, w])

        # Block 3: Depthwise separable convolution
        x = F.relu(self.block3_depthwise(x))
        x = F.relu(self.block3_pointwise(x))

        # Branch path: Average pooling
        branch = self.branch_avgpool(x)
        branch = paddle.reshape(branch, shape=[-1, 96, 32 * 32])

        # Main path: Flatten
        main = paddle.reshape(x, shape=[-1, 96 * 32 * 32])

        # Concatenate and classify
        combined = paddle.concat([main, branch], axis=1)
        out = self.fc(combined)

        return out

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected shape: [1, 10]