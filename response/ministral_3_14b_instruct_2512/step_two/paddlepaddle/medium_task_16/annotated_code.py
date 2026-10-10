import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Split channels into three groups and apply 1x1 convolutions
        self.conv1 = nn.Conv2D(in_channels='mask', out_channels='mask', kernel_size='mask', stride='mask', padding='mask')






        self.conv2 = nn.Conv2D(
            in_channels=3,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.conv3 = nn.Conv2D(
            in_channels=3,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # Average pooling for downsampling
        self.pool = nn.AvgPool2D(
            kernel_size=2,
            stride=2,
            padding=0
        )

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=1*32*32, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Split channels into three groups
        x1 = x[:, :1, :, :]  # output tensor shape: (1, 1, 32, 32)
        x2 = x[:, 1:2, :, :]  # output tensor shape: (1, 1, 32, 32)
        x3 = x[:, 2:, :, :]  # output tensor shape: (1, 1, 32, 32)

        # Apply 1x1 convolutions
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        # Downsample with average pooling
        x1 = self.pool(x1)
        x2 = self.pool(x2)
        x3 = self.pool(x3)

        # Concatenate along channel dimension
        x = paddle.concat([x1, x2, x3], axis=1)

        # Flatten and pass through fully connected layers
        x = paddle.flatten(x, start_axis=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn((1, 3, 32, 32))  # Batch of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print("Output shape:", output_tensor.shape)