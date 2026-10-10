import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Main Path
        self.conv2d_1 = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv2d_2 = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)

        # Branch Path
        self.conv2d_branch = nn.Conv2D(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)

        # Fully Connected Layer
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=3072, out_features=10)

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Split input along the last dimension
        x1 = x[:, :, :, :1]  # output tensor shape: (1, 3, 32, 1)
        x2 = x[:, :, :, 1:2]  # output tensor shape: (1, 3, 32, 1)
        x3 = x[:, :, :, 2:]  # output tensor shape: (1, 3, 32, 30)

        # Main Path
        x2 = self.conv2d_1(x2)  # output tensor shape: (1, 3, 32, 1)
        x_combined = paddle.concat([x2, x3], axis=3)  # output tensor shape: (1, 3, 32, 31)
        x_main = self.conv2d_2(x_combined)  # output tensor shape: (1, 3, 32, 31)
        x_main = paddle.concat([x1, x_main], axis=3)  # output tensor shape: (1, 3, 32, 32)

        # Branch Path
        x_branch = self.conv2d_branch(x)  # output tensor shape: (1, 3, 32, 32)

        # Fusion
        x_fused = paddle.add(x_main, x_branch)  # output tensor shape: (1, 3, 32, 32)

        # Classification
        x_flattened = self.flatten(x_fused)  # output tensor shape: (1, 3072)
        output = self.linear(x_flattened)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)