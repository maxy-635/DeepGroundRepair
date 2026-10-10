import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: Split channels and apply separable convolutions with different kernel sizes
        self.conv1x1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)

        # Batch norm and activation for first block
        self.bn1 = nn.BatchNorm2D(num_features=64)
        self.relu = nn.ReLU()

        # Second block: Multiple branches
        # Branch 1: 3x3 convolution
        self.branch1_conv = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Branch 2: 1x1 followed by two 3x3 convolutions
        self.branch2_conv1 = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch2_conv2 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.branch2_conv3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Branch 3: Max pooling
        self.branch3_pool = nn.MaxPool2D(kernel_size=3, stride=1, padding=1)

        # Batch norm and activation for second block
        self.bn2 = nn.BatchNorm2D(num_features=192)

        # Global average pooling and final classification layer
        self.avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))
        self.fc = nn.Linear(in_features=192, out_features=10)

    def forward(self, x):
        # First block processing
        conv1x1_out = self.conv1x1(x)
        conv3x3_out = self.conv3x3(x)
        conv5x5_out = self.conv5x5(x)

        # Concatenate outputs from first block
        block1_out = paddle.concat([conv1x1_out, conv3x3_out, conv5x5_out], axis=1)
        block1_out = self.bn1(block1_out)
        block1_out = self.relu(block1_out)

        # Second block processing
        # Branch 1
        branch1_out = self.branch1_conv(block1_out)

        # Branch 2
        branch2_out = self.branch2_conv1(block1_out)
        branch2_out = self.bn2(branch2_out)
        branch2_out = self.relu(branch2_out)
        branch2_out = self.branch2_conv2(branch2_out)
        branch2_out = self.bn2(branch2_out)
        branch2_out = self.relu(branch2_out)
        branch2_out = self.branch2_conv3(branch2_out)

        # Branch 3
        branch3_out = self.branch3_pool(block1_out)

        # Concatenate outputs from second block
        block2_out = paddle.concat([branch1_out, branch2_out, branch3_out], axis=1)

        # Global average pooling and final classification
        pooled_out = self.avg_pool(block2_out)
        pooled_out = paddle.flatten(pooled_out, start_axis=1)
        out = self.fc(pooled_out)

        return out

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, 3, 32, 32] for CIFAR-10
    input_tensor = paddle.randn((1, 3, 32, 32))
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)  # Should be [1, 10]