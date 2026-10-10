import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # First block: split into three groups with different kernel sizes
        self.conv1x1_1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv3x3_1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5x5_1 = nn.Conv2D(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2)

        # Second block: multiple branches with different configurations
        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        )

        # Branch 2: <1x1 convolution, 3x3 convolution>
        self.branch2_conv1x1 = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch2_conv3x3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Branch 3: <1x1 convolution, 3x3 convolution>
        self.branch3_conv1x1 = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.branch3_conv3x3 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Branch 4: <max pooling, 1x1 convolution>
        self.branch4_pool = nn.MaxPool2D(kernel_size=3, stride=1, padding=1)
        self.branch4_conv1x1 = nn.Conv2D(in_channels=192, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Layer normalization
        self.ln = nn.LayerNorm(normalized_shape=[256, 32, 32])

        # Final layers
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=64*32*32, out_features=10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # First block: split into three groups
        conv1x1_out = self.conv1x1_1(x)  # output tensor shape: (1, 64, 32, 32)
        conv3x3_out = self.conv3x3_1(x)  # output tensor shape: (1, 64, 32, 32)
        conv5x5_out = self.conv5x5_1(x)  # output tensor shape: (1, 64, 32, 32)

        # Concatenate along channel dimension
        first_block_out = paddle.concat([conv1x1_out, conv3x3_out, conv5x5_out], axis=1)  # output tensor shape: (1, 192, 32, 32)

        # Second block: multiple branches
        # Branch 1
        branch1_out = self.branch1(first_block_out)  # output tensor shape: (1, 64, 32, 32)

        # Branch 2
        branch2_conv1x1_out = self.branch2_conv1x1(first_block_out)  # output tensor shape: (1, 64, 32, 32)
        branch2_conv3x3_out = self.branch2_conv3x3(branch2_conv1x1_out)  # output tensor shape: (1, 64, 32, 32)
        branch2_out = branch2_conv3x3_out  # output tensor shape: (1, 64, 32, 32)

        # Branch 3
        branch3_conv1x1_out = self.branch3_conv1x1(first_block_out)  # output tensor shape: (1, 64, 32, 32)
        branch3_conv3x3_out = self.branch3_conv3x3(branch3_conv1x1_out)  # output tensor shape: (1, 64, 32, 32)
        branch3_out = branch3_conv3x3_out  # output tensor shape: (1, 64, 32, 32)

        # Branch 4
        branch4_pool_out = self.branch4_pool(first_block_out)  # output tensor shape: (1, 192, 32, 32)
        branch4_conv1x1_out = self.branch4_conv1x1(branch4_pool_out)  # output tensor shape: (1, 64, 32, 32)
        branch4_out = branch4_conv1x1_out  # output tensor shape: (1, 64, 32, 32)

        # Concatenate all branches
        second_block_out = paddle.concat([branch1_out, branch2_out, branch3_out, branch4_out], axis=1)  # output tensor shape: (1, 256, 32, 32)

        # Apply layer normalization
        normalized_out = self.ln(second_block_out)

        # Final layers
        flattened = self.flatten(normalized_out)
        output = self.fc(flattened)

        return output

if __name__ == "__main__":
    model = DL_Model()
    # Create a dummy input tensor with shape [batch_size, channels, height, width]
    # CIFAR-10 images are 32x32 with 3 channels
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)