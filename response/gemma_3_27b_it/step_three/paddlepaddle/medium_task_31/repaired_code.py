import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()
        # Split input into three groups along the channel dimension
        in_channels = 3  # output tensor shape: (1)
        self.conv1x1 = nn.Conv2D(in_channels=in_channels // 3, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2D(in_channels=in_channels // 3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5x5 = nn.Conv2D(in_channels=in_channels // 3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=98304, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Split the input tensor into three groups along the channel dimension
        x1 = x[:, :1, :, :]  # output tensor shape: (1, 1, 32, 32)
        x2 = x[:, 1:2, :, :]  # output tensor shape: (1, 1, 32, 32)
        x3 = x[:, 2:, :, :]  # output tensor shape: (1, 1, 32, 32)

        # Apply different convolutional kernels to each group
        x1 = self.conv1x1(x1)  # output tensor shape: (1, 32, 32, 32)
        x2 = self.conv3x3(x2)  # output tensor shape: (1, 32, 32, 32)
        x3 = self.conv5x5(x3)  # output tensor shape: (1, 32, 32, 32)

        # Concatenate the outputs from the three groups
        x = paddle.concat([x1, x2, x3], axis=1)  # output tensor shape: (1, 96, 32, 32)

        # Flatten the concatenated features
        x = self.flatten(x)  # output tensor shape: (1, 98304)

        # Pass the flattened features through two fully connected layers
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn(shape=[1, 3, 32, 32])
    output_tensor = model(input_tensor)
    print(output_tensor.shape)