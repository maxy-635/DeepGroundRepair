import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Block 1 - Primary path
        self.conv1x1_1 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.depthwise_conv_1 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, groups=32)
        self.conv1x1_2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Block 1 - Branch path
        self.depthwise_conv_2 = nn.Conv2D(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, groups=1)
        self.conv1x1_3 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)

        # Block 2 - Channel shuffling
        self.groups = 4

        # Final classification layer
        self.fc = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (32, 1, 28, 28)
        # Block 1 - Primary path
        x1 = F.relu(self.conv1x1_1(x))  # output tensor shape: (32, 32, 28, 28)
        x1 = F.relu(self.depthwise_conv_1(x1))  # output tensor shape: (32, 32, 28, 28)
        x1 = F.relu(self.conv1x1_2(x1))  # output tensor shape: (32, 64, 28, 28)

        # Block 1 - Branch path
        x2 = F.relu(self.depthwise_conv_2(x))  # output tensor shape: (32, 32, 28, 28)
        x2 = F.relu(self.conv1x1_3(x2))  # output tensor shape: (32, 64, 28, 28)

        # Concatenate along channel dimension
        x = paddle.concat([x1, x2], axis=1)  # output tensor shape: (32, 128, 28, 28)

        # Block 2 - Channel shuffling
        batch_size, channels, height, width = x.shape  # output tensor shape: (1); (1); (1); (1)
        channels_per_group = channels // self.groups  # output tensor shape: (1)

        # Reshape to (batch_size, height, width, groups, channels_per_group)
        x = x.reshape([batch_size, height, width, self.groups, channels_per_group])  # output tensor shape: (32, 28, 28, 4, 32)

        # Permute dimensions to (batch_size, height, width, channels_per_group, groups)
        x = x.transpose([0, 1, 2, 4, 3])  # output tensor shape: (32, 28, 28, 32, 4)

        # Reshape back to original shape
        x = x.reshape([batch_size, channels, height, width])  # output tensor shape: (32, 128, 28, 28)

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))  # output tensor shape: (32, 128, 1, 1)

        # Flatten and pass through fully connected layer
        x = paddle.flatten(x, start_axis=1)  # output tensor shape: (32, 128)
        x = self.fc(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([32, 1, 28, 28])  # Batch of 32 MNIST images (1 channel, 28x28)
    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)