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
        self.linear = nn.Linear(in_features=3 * 3 * 32, out_features=10)  # Assuming input image size is 32x32

    def forward(self, x):
        # Split input along the last dimension
        x1 = x[:, :, :, :1]
        x2 = x[:, :, :, 1:2]
        x3 = x[:, :, :, 2:]

        # Main Path
        x2 = self.conv2d_1(x2)
        x_combined = paddle.concat([x2, x3], axis=3)
        x_main = self.conv2d_2(x_combined)
        x_main = paddle.concat([x1, x_main], axis=3)

        # Branch Path
        x_branch = self.conv2d_branch(x)

        # Fusion
        x_fused = paddle.add(x_main, x_branch)

        # Classification
        x_flattened = self.flatten(x_fused)
        output = self.linear(x_flattened)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)