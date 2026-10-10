import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Assuming input channels are 3 (for CIFAR-10)
        input_channels = 3  # output tensor shape: (1)
        # Split into 3 groups, each with 1 channel
        group_channels = input_channels // 3  # output tensor shape: (1)

        # 1x1 convolutions for each group
        self.conv1 = nn.Conv2D(
            in_channels=group_channels,
            out_channels=group_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.conv2 = nn.Conv2D(
            in_channels=group_channels,
            out_channels=group_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.conv3 = nn.Conv2D(
            in_channels=group_channels,
            out_channels=group_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Average pooling for each group
        self.pool = nn.AvgPool2D(
            kernel_size=2,
            stride=2,
            padding=0
        )

        # Flatten and fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(



        self.fc2 = nn.Linear(
            in_features=128,
            out_features=10
        )

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Split the input into 3 groups along the channel dimension
        x1, x2, x3 = paddle.split(x, num_or_sections=3, axis=1)  # output tensor shape: (1, 1, 32, 32); (1, 1, 32, 32); (1, 1, 32, 32)

        # Apply 1x1 convolutions to each group
        x1 = self.conv1(x1)  # output tensor shape: (1, 1, 32, 32)
        x2 = self.conv2(x2)  # output tensor shape: (1, 1, 32, 32)
        x3 = self.conv3(x3)  # output tensor shape: (1, 1, 32, 32)

        # Downsample each group via average pooling
        x1 = self.pool(x1)  # output tensor shape: (1, 1, 16, 16)
        x2 = self.pool(x2)  # output tensor shape: (1, 1, 16, 16)
        x3 = self.pool(x3)  # output tensor shape: (1, 1, 16, 16)

        # Concatenate the feature maps along the channel dimension
        x = paddle.concat([x1, x2, x3], axis=1)  # output tensor shape: (1, 3, 16, 16)

        # Flatten and pass through fully connected layers
        x = self.flatten(x)  # output tensor shape: (1, 768)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]