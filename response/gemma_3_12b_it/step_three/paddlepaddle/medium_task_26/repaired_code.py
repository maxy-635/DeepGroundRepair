import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # 1x1 convolution to compress channels
        self.conv1x1 = nn.Conv2D(
            in_channels=64,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )
        # Parallel convolutional layers
        self.conv1x1_expand = nn.Conv2D(
            in_channels=64,
            out_channels=128,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.conv3x3_expand = nn.Conv2D(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )
        # Flatten the feature map
        self.flatten = nn.Flatten()
        # Fully connected layers
        self.linear1 = nn.Linear(in_features=262144, out_features=64)



        self.linear2 = nn.Linear(
            in_features=64,
            out_features=10
        )

    def forward(self, x):
        # the initial shape of x is: (1, 64, 32, 32)
        # Compress channels with 1x1 convolution
        x = self.conv1x1(x)  # output tensor shape: (1, 64, 32, 32)
        # Expand features with parallel convolutions
        x1 = self.conv1x1_expand(x)  # output tensor shape: (1, 128, 32, 32)
        x2 = self.conv3x3_expand(x)  # output tensor shape: (1, 128, 32, 32)
        # Concatenate the results
        x = paddle.concat([x1, x2], axis=1)  # output tensor shape: (1, 256, 32, 32)
        # Flatten the feature map
        x = self.flatten(x)  # output tensor shape: (1, 262144)
        # Pass through fully connected layers
        x = self.linear1(x)
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    model = DL_Model()
    # Dummy input tensor
    input_tensor = paddle.rand(shape=(1, 64, 32, 32))
    output_tensor = model(input_tensor)
    print(output_tensor.shape)