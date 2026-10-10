import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self, num_classes=10):
        super(DL_Model, self).__init__()

        # First block: Split into three groups and apply separable convolutions
        self.conv1x1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)

        # Second block: Multiple branches for enhanced feature extraction
        # Branch 1: 3x3 convolution
        self.branch1_conv = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Branch 2: 1x1 convolution followed by two 3x3 convolutions
        self.branch2_conv1 = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch2_conv2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch2_conv3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Branch 3: Max pooling
        self.branch3_pool = nn.MaxPool2D(kernel_size=3, stride=1, padding=1)

        # Global average pooling and fully connected layer
        self.global_avg_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.fc = nn.Linear(in_features=192, out_features=num_classes)

    def forward(self, x):
        # First block: Split and apply separable convolutions
        x_split = paddle.split(x, num_or_sections=3, axis=1)
        x1 = self.conv1x1(x_split[0])
        x2 = self.conv3x3(x_split[1])
        x3 = self.conv5x5(x_split[2])
        x_concat = paddle.concat([x1, x2, x3], axis=1)

        # Second block: Multiple branches
        # Branch 1
        branch1 = self.branch1_conv(x_concat)

        # Branch 2
        branch2 = self.branch2_conv1(x_concat)
        branch2 = self.branch2_conv2(branch2)
        branch2 = self.branch2_conv3(branch2)

        # Branch 3
        branch3 = self.branch3_pool(x_concat)

        # Concatenate all branches
        x_concat = paddle.concat([branch1, branch2, branch3], axis=1)

        # Global average pooling and fully connected layer
        x_pool = self.global_avg_pool(x_concat)
        x_pool = paddle.reshape(x_pool, shape=[x_pool.shape[0], -1])
        output = self.fc(x_pool)

        return output

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])  # CIFAR-10 input shape
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)