import paddle
import paddle.nn as nn

class DL_Model(nn.Layer):
    def __init__(self):
        super(DL_Model, self).__init__()

        # Path 1
        self.path1_conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.path1_conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.path1_pool = nn.AvgPool2D(kernel_size=2, stride=2, padding=0)

        # Path 2
        self.path2_conv = nn.Conv2D(in_channels=3, out_channels='mask', kernel_size='mask', stride='mask', padding='mask')

        # Flatten and Linear layer
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=64 * 8 * 8 + 32 * 32 * 32, out_features=10)  # Assuming input image size is 32x32

    def forward(self, x):
        # the initial shape of x is: (1, 3, 32, 32)
        # Path 1
        path1 = self.path1_conv1(x)  # output tensor shape: (1, 32, 32, 32)
        path1 = self.path1_conv2(path1)  # output tensor shape: (1, 64, 32, 32)
        path1 = self.path1_pool(path1)  # output tensor shape: (1, 64, 16, 16)

        # Path 2
        path2 = self.path2_conv(x)  # output tensor shape of this layer should be (1, 64, 16, 16)

        # Combine paths
        combined = path1 + path2

        # Flatten and Linear layer
        flattened = self.flatten(combined)
        output = self.linear(flattened)

        return output


if __name__ == "__main__":
    model = DL_Model()
    input_tensor = paddle.randn([1, 3, 32, 32])  # Batch size of 1, 3 channels, 32x32 image
    output_tensor = model(input_tensor)
    print(output_tensor.shape)  # Expected output shape: [1, 10]